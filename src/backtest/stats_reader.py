import os
import glob
import csv
import logging
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

class BacktestStatsReader:
    """
    Parses and caches historical backtest results across all tickers.
    Provides instant in-memory lookup of empirical win rates, average returns, 
    and strategy edge scores without making any external API calls.
    """
    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = results_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../backtest_results")
        )
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._last_loaded_file: Optional[str] = None
        self.reload_latest_results()

    def reload_latest_results(self) -> bool:
        """Finds and parses the newest backtest CSV in the results directory."""
        if not os.path.exists(self.results_dir):
            log.warning(f"Backtest results directory not found at {self.results_dir}")
            return False

        csv_files = glob.glob(os.path.join(self.results_dir, "*.csv"))
        if not csv_files:
            log.info("No historical backtest CSV files found.")
            return False

        latest_file = max(csv_files, key=os.path.getmtime)
        if latest_file == self._last_loaded_file and self._stats_cache:
            return True

        try:
            trades_by_ticker: Dict[str, list] = {}
            with open(latest_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ticker = row.get("ticker", "").upper()
                    if ticker:
                        if ticker not in trades_by_ticker:
                            trades_by_ticker[ticker] = []
                        trades_by_ticker[ticker].append(row)

            new_cache = {}
            for ticker, trades in trades_by_ticker.items():
                total = len(trades)
                wins = sum(1 for t in trades if t.get("outcome") == "WIN")
                losses = sum(1 for t in trades if t.get("outcome") == "LOSS")
                decided = wins + losses
                win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
                
                returns = []
                max_gains = []
                days_list = []
                for t in trades:
                    try:
                        returns.append(float(t.get("option_pnl_pct", 0.0)))
                        max_gains.append(float(t.get("max_gain_pct", 0.0)))
                        days_list.append(float(t.get("days_held", 1.0)))
                    except (ValueError, TypeError):
                        continue

                avg_return = sum(returns) / len(returns) if returns else 0.0
                max_gain = max(max_gains) if max_gains else 0.0
                avg_hold = sum(days_list) / len(days_list) if days_list else 1.0
                asset_tier = trades[0].get("asset_tier", "MID-CAP")

                new_cache[ticker] = {
                    "ticker": ticker,
                    "asset_tier": asset_tier,
                    "total_trades": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(win_rate, 1),
                    "avg_return_pct": round(avg_return, 1),
                    "max_gain_pct": round(max_gain, 1),
                    "avg_days_held": round(avg_hold, 1),
                    "has_edge": win_rate >= 50.0 and avg_return > 0.0
                }

            self._stats_cache = new_cache
            self._last_loaded_file = latest_file
            log.info(f"Loaded historical backtest statistics for {len(new_cache)} tickers from {os.path.basename(latest_file)}")
            return True

        except Exception as e:
            log.error(f"Failed to parse backtest CSV {latest_file}: {e}")
            return False

    def get_ticker_stats(self, ticker: str) -> Dict[str, Any]:
        """Returns the empirical backtest performance dictionary for a specific ticker."""
        self.reload_latest_results()
        sym = ticker.upper()
        if sym in self._stats_cache:
            return self._stats_cache[sym]
        
        # Fallback if ticker has not been backtested yet
        return {
            "ticker": sym,
            "asset_tier": "UNKNOWN",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "max_gain_pct": 0.0,
            "avg_days_held": 0.0,
            "has_edge": False
        }
