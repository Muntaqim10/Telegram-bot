import logging
import json
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import asyncio
from src.database import close_trade as db_close_trade

log = logging.getLogger(__name__)

# Open option positions are multi-day (14-21 DTE) instruments, so they have to
# outlive both the daily reset and a process restart -- otherwise nothing is left
# watching them when expiration arrives.
DEFAULT_STATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/active_positions.json"))

# Thresholds derived from this account's own 2019-2026 Robinhood history:
#   hold 0 days  -> 50% win rate, +24.3% avg return, +$5,263 net
#   hold 1 day   -> 53% win rate,  +4.6% avg return, +$7,329 net
#   hold 2-5 days-> 44% win rate,  +6.1% avg return, +$6,077 net
#   hold 6+ days -> 27% win rate, -20.7% avg return, -$1,987 net   <-- the leak
# and 69 closed contracts were sold only after losing 50-99% of premium.
DEFAULT_MAX_HOLD_DAYS = 5
DEFAULT_PREMIUM_STOP_PCT = -0.50

# A position whose last observed tick is older than this is not valued at all: the
# streaming universe rotates, so a held ticker can silently stop receiving ticks, and
# a stop decision made on a stale price is worse than no decision.
PRICE_STALE_AFTER_SECONDS = 3600

# The outcome log's columns. Kept as a constant because the header is only written when
# the file is first created: when a column was added later, existing files kept a stale
# 7-column header while new rows carried 8 fields, which made the whole log unparseable
# by pandas and silently unusable for calibration.
OUTCOME_CSV_COLUMNS = ["timestamp", "ticker", "direction", "entry_price", "exit_price",
                       "outcome", "pnl_pct", "estimated_option_pnl_pct"]
_outcome_header_checked = False

class RiskManager:
    """
    Handles intraday risk management, specifically focusing on trailing stops
    that give >5% momentum runners room to breathe during normal intraday pullbacks.
    """
    def __init__(self, atr_multiplier: float = 2.5, min_profit_pct: float = 0.10, state_path: str = DEFAULT_STATE_PATH,
                 max_hold_days: int = DEFAULT_MAX_HOLD_DAYS, premium_stop_pct: float = DEFAULT_PREMIUM_STOP_PCT):
        self.atr_multiplier = atr_multiplier
        self.min_profit_pct = min_profit_pct
        self.max_hold_days = max_hold_days
        self.premium_stop_pct = premium_stop_pct
        self.active_positions = {}
        self.pending_evaluations = set()
        self._state_path = state_path  # set to None to disable on-disk persistence
        
    def mark_pending(self, ticker: str) -> bool:
        """Atomically mark a ticker as currently evaluating to prevent double-entry race conditions."""
        if ticker in self.active_positions or ticker in self.pending_evaluations:
            return False
        self.pending_evaluations.add(ticker)
        return True
        
    def clear_pending(self, ticker: str):
        """Release the evaluation lock for a ticker."""
        self.pending_evaluations.discard(ticker)

    @staticmethod
    def market_today() -> str:
        """Today's date in market time. Every date in this class is US/Eastern so that
        entry dates, hold counts and expiration checks cannot disagree on a UTC host."""
        return pd.Timestamp.now(tz="US/Eastern").strftime("%Y-%m-%d")

    def days_held(self, pos: dict, current_date: str = None):
        """Calendar days the position has been open, or None if the entry date is
        missing or unparseable. Calendar days (not trading days) to match the basis
        the max_hold_days threshold was measured on."""
        entry_date = pos.get("entry_date")
        if not entry_date:
            return None
        try:
            cur = datetime.strptime(str(current_date or self.market_today()), "%Y-%m-%d").date()
            ent = datetime.strptime(str(entry_date), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None
        return (cur - ent).days

    def _estimate_option_pnl(self, pos: dict, current_price: float, current_date: str = None):
        """Approximates the option's return from entry delta plus elapsed theta decay.

        ESTIMATE ONLY: linear in the underlying, no gamma, no vega/IV change, no live
        option quote. Theta matters here because the alerts that consume this fire on
        positions held for days -- ignoring it made the estimate optimistic by exactly
        the amount that had decayed away. Returns None when there is no option data to
        work from, rather than fabricating a number.
        """
        opt_entry = pos.get("option_entry_price")
        opt_delta = pos.get("option_entry_delta")
        if not (opt_entry and opt_entry > 0 and opt_delta is not None and opt_delta != 0):
            return None

        stock_move = current_price - pos["entry_price"]
        estimated_move = stock_move * abs(opt_delta)
        if pos.get("direction") == "Short":
            estimated_move = -stock_move * abs(opt_delta)

        # Theta is per-day and negative for long premium; clamp so a bad feed can only
        # ever decay the estimate, never inflate it.
        theta = pos.get("option_entry_theta")
        held = self.days_held(pos, current_date)
        decay = 0.0
        if theta is not None and held:
            decay = min(0.0, float(theta)) * max(0, held)

        estimated_price = max(0.01, opt_entry + estimated_move + decay)
        return (estimated_price - opt_entry) / opt_entry

    def _price_is_fresh(self, pos: dict) -> bool:
        """True when the last observed tick is recent enough to value the position."""
        ts = pos.get("last_price_ts")
        if not ts:
            return False
        try:
            return (time.time() - float(ts)) <= PRICE_STALE_AFTER_SECONDS
        except (TypeError, ValueError):
            return False

    def _persist(self) -> None:
        """Writes active positions to disk so a restart does not lose track of open
        contracts. Best-effort by design: a disk problem must never interrupt trading."""
        if not self._state_path:
            return
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            tmp_path = self._state_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.active_positions, f, indent=2)
            os.replace(tmp_path, self._state_path)  # atomic swap; never a half-written file
        except Exception as e:
            log.error(f"Failed to persist active positions: {e}")

    def load_positions(self, current_date: str = None) -> int:
        """Restores positions saved by a previous run, dropping any whose option has
        already expired. Returns the number of positions restored."""
        if self.active_positions:
            log.warning(
                f"load_positions() called with {len(self.active_positions)} live position(s) already "
                f"in memory; refusing to clobber them."
            )
            return 0
        if not self._state_path or not os.path.exists(self._state_path):
            return 0
        try:
            with open(self._state_path, encoding="utf-8") as f:
                saved = json.load(f)
        except Exception as e:
            log.error(f"Failed to load persisted positions: {e}")
            return 0

        if not isinstance(saved, dict):
            log.error("Persisted position state is not a mapping; ignoring it.")
            return 0

        restored = {}
        for ticker, pos in saved.items():
            if not isinstance(pos, dict) or "entry_price" not in pos or "direction" not in pos:
                log.warning(f"Skipping malformed persisted position for {ticker}.")
                continue
            if current_date is not None:
                days_left = self._days_to_expiration(pos, current_date)
                if days_left is not None and days_left < 0:
                    log.info(f"[{ticker}] Persisted position discarded: option expired {pos.get('option_expiration')}.")
                    continue
            restored[ticker] = pos

        self.active_positions = restored
        if restored:
            log.info(f"Restored {len(restored)} active position(s) from disk: {', '.join(sorted(restored))}")
        self._persist()
        return len(restored)

    def requeue_warning(self, ticker: str, flag: str = "expiration_warned") -> None:
        """Hands a consumed warning back so the next sweep retries it. Called when a
        Telegram dispatch fails -- without it, one failed send silently swallows the
        only alert that position will ever produce."""
        pos = self.active_positions.get(ticker)
        if pos is not None:
            pos[flag] = False
            self._persist()

    def confirm_position(self, ticker: str, quantity: float = None, fill_price: float = None,
                         entry_date: str = None) -> dict:
        """Promotes a tracked alert into a real position the trader actually took.

        Only confirmed positions receive expiration and exit alerts. Without this the bot
        would nag about contracts that were never bought, and the trader would learn to
        ignore the alerts -- including the one that matters.
        """
        pos = self.active_positions.get(ticker)
        if pos is None:
            return None
        pos["confirmed"] = True
        pos["entry_date"] = entry_date or self.market_today()
        if quantity is not None:
            pos["quantity"] = quantity
        if fill_price is not None:
            pos["fill_price"] = fill_price
            # The trader's real fill beats the quote the pipeline saw at alert time.
            pos["option_entry_price"] = fill_price
        # A freshly confirmed position starts its alert budget over.
        pos["expiration_warned"] = False
        pos["time_stop_warned"] = False
        pos["premium_stop_warned"] = False
        self._persist()
        log.info(f"[{ticker}] Position CONFIRMED by trader"
                 f"{f' ({quantity} @ {fill_price})' if fill_price else ''}.")
        return pos

    def close_confirmed(self, ticker: str, exit_option_price: float = None) -> dict:
        """Closes a confirmed position and records the REAL option P&L when given.

        This is the only path by which a genuine fill reaches the outcome log; every
        other number in it is a delta+theta approximation.
        """
        pos = self.active_positions.get(ticker)
        if pos is None:
            return None

        entry = pos["entry_price"]
        stock_exit = pos.get("last_price", entry)
        stock_pnl = ((stock_exit - entry) / entry) if pos.get("direction") == "Long" else ((entry - stock_exit) / entry)

        real_option_pnl = None
        opt_entry = pos.get("option_entry_price")
        if exit_option_price is not None and opt_entry and opt_entry > 0:
            real_option_pnl = (exit_option_price - opt_entry) / opt_entry

        summary = {
            "ticker": ticker,
            "direction": pos.get("direction"),
            "quantity": pos.get("quantity"),
            "option_entry_price": opt_entry,
            "exit_option_price": exit_option_price,
            "option_pnl_pct": real_option_pnl,
            "days_held": self.days_held(pos),
            "was_confirmed": bool(pos.get("confirmed")),
        }
        self.close_trade(ticker, "MANUAL_CLOSE", stock_exit, stock_pnl, real_option_pnl)
        return summary

    def list_positions(self) -> dict:
        """Splits what the trader actually holds from what was merely alerted."""
        confirmed, alerts_only = [], []
        for ticker, pos in self.active_positions.items():
            (confirmed if pos.get("confirmed") else alerts_only).append((ticker, pos))
        return {"confirmed": confirmed, "unconfirmed": alerts_only}

    def set_entry_date(self, ticker: str, entry_date: str = None) -> None:
        """Records when the position was actually entered, so the hold-time check has a
        reference point. Defaults to the market's date rather than the host's, so a UTC
        server does not record tomorrow for an evening signal."""
        pos = self.active_positions.get(ticker)
        if pos is not None:
            pos["entry_date"] = entry_date or self.market_today()
            self._persist()

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
        pos["last_price"] = current_price  # lets the hourly health sweep value the position
        pos["last_price_ts"] = time.time()  # ...but only while it is demonstrably fresh
        entry_price = pos["entry_price"]
        current_stop = pos["trailing_stop"]
        take_profit = pos["take_profit"]
        
        atr_distance = current_atr * self.atr_multiplier if current_atr > 0 else 0
        outcome = None
        
        # Track favorable movement
        was_favorable = pos.get("has_moved_favorably", False)
        if direction == "Long" and current_price > entry_price:
            pos["has_moved_favorably"] = True
        elif direction == "Short" and current_price < entry_price:
            pos["has_moved_favorably"] = True

        # Check thesis invalidation FIRST (Early exit if price reclaims/loses original breakout level after moving favorably)
        invalidation_level = pos.get("invalidation_level")
        if invalidation_level is not None and pos.get("has_moved_favorably", False):
            if direction == "Long" and current_price < invalidation_level:
                profit_pct = (current_price - entry_price) / entry_price
                outcome = {"status": "INVALIDATED", "exit_price": current_price, "pnl": profit_pct}
                log.info(f"[{ticker}] THESIS INVALIDATED: Price (${current_price:.2f}) fell back below breakout level (${invalidation_level:.2f}). Exiting early.")
            elif direction == "Short" and current_price > invalidation_level:
                profit_pct = (entry_price - current_price) / entry_price
                outcome = {"status": "INVALIDATED", "exit_price": current_price, "pnl": profit_pct}
                log.info(f"[{ticker}] THESIS INVALIDATED: Price (${current_price:.2f}) rose back above breakdown level (${invalidation_level:.2f}). Exiting early.")

        if outcome is None:
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
            outcome["estimated_option_pnl_pct"] = self._estimate_option_pnl(pos, current_price)  # theta-aware
            return outcome
                
        # Persist only when the position's durable state actually moved -- writing on
        # every tick would put a disk write in the hot path.
        if pos["trailing_stop"] != current_stop or pos.get("has_moved_favorably", False) != was_favorable:
            self._persist()

        return {"status": "ACTIVE", "current_stop": pos["trailing_stop"]}

    def attach_option_pricing(self, ticker: str, option_entry_price: float, option_entry_delta: float, option_entry_theta: float, option_expiration: str = None) -> None:
        """Attaches real option entry data to an already-registered position.

        option_expiration is attached here rather than in add_position() because the
        pipeline registers the position before it resolves the option chain -- this is
        the first point in the flow where the contract's expiration date is known.
        """
        if ticker in self.active_positions:
            self.active_positions[ticker]["option_entry_price"] = option_entry_price
            self.active_positions[ticker]["option_entry_delta"] = option_entry_delta
            self.active_positions[ticker]["option_entry_theta"] = option_entry_theta
            if option_expiration:
                self.active_positions[ticker]["option_expiration"] = option_expiration
            self._persist()

    @staticmethod
    def _repair_outcome_csv(file_path: str) -> None:
        """Brings an existing outcome log up to the current column set.

        Runs once per process. A header written before a column was added stays stale
        forever, so rows drift wider than the header and pandas refuses to read the file
        at all -- which is how 528 outcome records became unusable without anyone noticing.
        """
        global _outcome_header_checked
        if _outcome_header_checked or not os.path.exists(file_path):
            return
        _outcome_header_checked = True

        import csv as _csv
        try:
            with open(file_path, newline="", encoding="utf-8") as f:
                rows = list(_csv.reader(f))
        except Exception as e:
            log.error(f"Could not read outcome log for repair: {e}")
            return

        if not rows or rows[0] == OUTCOME_CSV_COLUMNS:
            return

        width = len(OUTCOME_CSV_COLUMNS)
        body = rows[1:] if rows[0] and rows[0][0] == "timestamp" else rows
        padded = [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in body]
        try:
            tmp = file_path + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(OUTCOME_CSV_COLUMNS)
                w.writerows(padded)
            os.replace(tmp, file_path)
            log.warning(f"Repaired outcome log header and realigned {len(padded)} row(s) to {width} columns.")
        except Exception as e:
            log.error(f"Could not repair outcome log: {e}")

    def close_trade(self, ticker: str, outcome_status: str, exit_price: float, pnl_pct: float, estimated_option_pnl_pct: float = None):
        """
        Removes the active position and logs the final outcome to CSV for ML training.
        """
        if ticker not in self.active_positions:
            return

        pos = self.active_positions.pop(ticker)
        self._persist()
        
        # Log to file
        import os
        import csv
        
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/trade_outcomes.csv"))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self._repair_outcome_csv(file_path)
        file_exists = os.path.exists(file_path)
        
        opt_pnl_str = f"{estimated_option_pnl_pct:.4f}" if estimated_option_pnl_pct is not None else ""
        
        try:
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(OUTCOME_CSV_COLUMNS)
                
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

    def remove_position(self, ticker: str) -> None:
        """Removes a position WITHOUT logging an outcome -- used to roll 
        back a position that was speculatively added but never actually 
        alerted (e.g. suppressed by a downstream gate before the alert 
        was sent). This is distinct from close_trade(), which logs a 
        real outcome for a position that was genuinely tracked."""
        self.active_positions.pop(ticker, None)
        self._persist()

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
        option_entry_theta: float = None,
        invalidation_level: float = None,
        catalyst_type: str = "Standard Breakout",
        option_expiration: str = None,
        entry_date: str = None,
        confirmed: bool = False
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
            "invalidation_level": invalidation_level,
            "has_moved_favorably": False,
            "catalyst_type": catalyst_type,
            "option_expiration": option_expiration,
            "expiration_warned": False,
            "entry_date": entry_date,
            "last_price": entry_price,
            "last_price_ts": time.time(),
            "time_stop_warned": False,
            "premium_stop_warned": False,
            # An alert is a suggestion, not a holding. Nothing here is a real position
            # until the trader confirms it -- the bot cannot see the brokerage account.
            "confirmed": confirmed,
            "quantity": None,
            "fill_price": None
        }
        self._persist()
        inval_str = f", Invalidation: {invalidation_level:.2f}" if invalidation_level is not None else ""
        log.info(f"Position added for {ticker} ({direction}) at {entry_price}. Stop: {initial_stop:.2f}, TP: {tp:.2f}{inval_str}")
        return {"stop_loss": initial_stop, "take_profit": tp}
        
    def _days_to_expiration(self, pos: dict, current_date: str):
        """Calendar days from current_date to the position's option expiration.

        Both dates use the "%Y-%m-%d" convention used everywhere else in the pipeline.
        Returns None when the position carries no usable expiration date.
        """
        exp_str = pos.get("option_expiration")
        if not exp_str:
            return None
        try:
            exp_dt = datetime.strptime(str(exp_str), "%Y-%m-%d").date()
            cur_dt = datetime.strptime(str(current_date), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            log.warning(f"Unparseable expiration '{exp_str}' or current date '{current_date}'; skipping expiration check.")
            return None
        return (exp_dt - cur_dt).days

    def check_expiration_warnings(self, current_date: str) -> list[dict]:
        """Returns a list of {ticker, days_to_expiration} for all active positions
        whose option_expiration is within 2 days of current_date and haven't already
        been warned about.

        Each position fires at most one warning; reset_daily() clears the flag so a
        position still open the next day warns again.
        """
        warnings = []
        for ticker, pos in self.active_positions.items():
            # Unconfirmed alerts are suggestions, not holdings; never nag about them.
            if not pos.get("confirmed"):
                continue
            if not pos.get("option_expiration") or pos.get("expiration_warned"):
                continue
            days_left = self._days_to_expiration(pos, current_date)
            if days_left is None:
                continue
            if 0 <= days_left <= 2:
                warnings.append({"ticker": ticker, "days_to_expiration": days_left})
                pos["expiration_warned"] = True
        if warnings:
            self._persist()
        return warnings

    def check_position_health(self, current_date: str) -> list[dict]:
        """Returns alerts for positions exhibiting the two exit failures visible in this
        account's trade history: holding a long option past the point where the edge
        historically inverts, and riding a losing contract down instead of cutting it.

        Each alert fires at most once per position per day (reset_daily clears the flags).
        Returns a list of {ticker, reason, days_held, est_option_pnl_pct, detail}.
        """
        alerts = []
        for ticker, pos in self.active_positions.items():
            if not pos.get("confirmed"):
                continue
            held = self.days_held(pos, current_date)

            # Only value the position when a recent tick backs the price. A rotated-out
            # ticker stops streaming, and a stop fired on a days-old price is worse than
            # no stop at all.
            est = None
            if self._price_is_fresh(pos):
                est = self._estimate_option_pnl(pos, pos.get("last_price", pos["entry_price"]), current_date)
            elif pos.get("option_entry_price"):
                log.warning(f"[{ticker}] Position not valued this sweep: no tick within "
                            f"{PRICE_STALE_AFTER_SECONDS}s. Premium stop cannot evaluate.")

            if held is not None and held >= self.max_hold_days and not pos.get("time_stop_warned"):
                pos["time_stop_warned"] = True
                alerts.append({
                    "ticker": ticker,
                    "reason": "TIME_STOP",
                    "days_held": held,
                    "est_option_pnl_pct": est,
                    "detail": (f"held {held} day(s), past the {self.max_hold_days}-day mark where this "
                               f"account's win rate historically drops from ~50% to 27%"),
                })

            if est is not None and est <= self.premium_stop_pct and not pos.get("premium_stop_warned"):
                pos["premium_stop_warned"] = True
                alerts.append({
                    "ticker": ticker,
                    "reason": "PREMIUM_STOP",
                    "days_held": held,
                    "est_option_pnl_pct": est,
                    "detail": (f"estimated premium down {est*100:.0f}%, past the "
                               f"{self.premium_stop_pct*100:.0f}% line"),
                })

        if alerts:
            self._persist()
        return alerts

    def reset_daily(self, current_date: str = None):
        """Clears stale intraday runners so they do not leak into the next trading day.

        Positions holding an option that has NOT yet expired are deliberately kept: the
        contracts are 14-21 DTE instruments, and wiping them every morning is precisely
        what let 21 positions expire worthless with nobody watching. Kept positions have
        their expiration_warned flag cleared so they can warn again today.

        Dropped: positions with no option expiration (pure intraday runners, the original
        behaviour) and positions whose option has already expired.

        Without current_date the original clear-everything behaviour applies, so callers
        that have no notion of "today" are unaffected.
        """
        if current_date is None:
            self.active_positions.clear()
            self._persist()
            return

        kept = {}
        for ticker, pos in self.active_positions.items():
            if not pos.get("confirmed"):
                # Never taken (or never confirmed). Bounded to one day, as before.
                continue
            days_left = self._days_to_expiration(pos, current_date)
            if days_left is None:
                if pos.get("option_expiration"):
                    # Recorded but unreadable. Dropping it here would abandon a real open
                    # contract silently -- exactly the failure this feature exists to stop.
                    log.error(
                        f"[{ticker}] KEEPING position across the reset: expiration "
                        f"{pos.get('option_expiration')!r} could not be parsed. Expiration warnings "
                        f"cannot fire for it until the date is fixed."
                    )
                    pos["expiration_warned"] = False
                    pos["time_stop_warned"] = False
                    pos["premium_stop_warned"] = False
                    kept[ticker] = pos
                else:
                    log.info(f"[{ticker}] Intraday position cleared by daily reset (no option expiration tracked).")
                continue
            if days_left < 0:
                log.info(f"[{ticker}] Position dropped: option expired on {pos.get('option_expiration')}.")
                continue
            pos["expiration_warned"] = False
            pos["time_stop_warned"] = False
            pos["premium_stop_warned"] = False
            kept[ticker] = pos

        dropped = len(self.active_positions) - len(kept)
        self.active_positions = kept
        self._persist()
        if kept:
            log.info(
                f"🔄 [DAILY RESET] Cleared {dropped} unconfirmed/intraday/expired entr(ies); "
                f"carrying {len(kept)} confirmed position(s) forward: {', '.join(sorted(kept))}"
            )
        elif dropped:
            log.info(f"🔄 [DAILY RESET] Cleared {dropped} unconfirmed/intraday/expired entr(ies); nothing held.")

