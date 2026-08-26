import os
import csv
import sys
import math
import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.strategy.donchian_daily import DonchianSwingStrategy
from src.data.dynamic_scanner import get_asset_tier_info
from src.utils.math_utils import black_scholes_price, black_scholes_delta, black_scholes_theta

log = logging.getLogger(__name__)

class BacktestEngine:
    """
    Headless Backtesting Engine with Black-Scholes 0.75 Delta ITM Options Modeling.
    Downloads historical daily bars from Tradier, runs the Donchian strategy,
    simulates realistic In-The-Money (~0.75 Delta) option pricing with daily Theta decay,
    and outputs signal outcomes & option ROI to CSV.
    """
    def __init__(self, tradier_token: str):
        self.tradier_token = tradier_token
        self.results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest_results"))
        os.makedirs(self.results_dir, exist_ok=True)
        
    async def fetch_historical_daily(self, session: aiohttp.ClientSession, ticker: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical daily bars from Tradier for backtesting."""
        headers = {"Authorization": f"Bearer {self.tradier_token}", "Accept": "application/json"}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        url = "https://api.tradier.com/v1/markets/history"
        params = {
            "symbol": ticker,
            "interval": "daily",
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d")
        }
        
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if not data or not isinstance(data, dict):
                        return pd.DataFrame()
                    history = data.get("history") or {}
                    days_data = history.get("day", []) if isinstance(history, dict) else []
                    if not isinstance(days_data, list):
                        days_data = [days_data]
                    if not days_data or not days_data[0]:
                        return pd.DataFrame()
                    df = pd.DataFrame(days_data)
                    if not df.empty and "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df.set_index("date", inplace=True)
                        for col in ["open", "high", "low", "close", "volume"]:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                        return df.dropna(subset=["close"])
        except Exception as e:
            log.warning(f"Error fetching daily history for {ticker}: {e}")
        return pd.DataFrame()

    @staticmethod
    def snap_to_exchange_strike(price: float) -> float:
        """Snaps mathematical strike to realistic standard US exchange strike increments."""
        if price < 25.0:
            return round(price * 2.0) / 2.0  # $0.50 increments
        elif price < 100.0:
            return round(price)              # $1.00 increments
        elif price < 250.0:
            return round(price / 2.5) * 2.5  # $2.50 increments
        else:
            return round(price / 5.0) * 5.0  # $5.00 increments

    def find_075_delta_strike(self, spot: float, direction: str, sigma: float, dte_days: int = 18) -> float:
        """Finds the exchange strike price closest to 0.75 Delta using Black-Scholes."""
        T = max(1.0, dte_days) / 365.0
        opt_type = "call" if direction == "Long" else "put"
        
        best_strike = self.snap_to_exchange_strike(spot)
        best_diff = 999.0
        
        # Test strikes from 25% ITM to 2% OTM in increments
        for pct in np.linspace(-0.25, 0.05, 80):
            raw_strike = spot * (1.0 + pct) if opt_type == "call" else spot * (1.0 - pct)
            strike = self.snap_to_exchange_strike(raw_strike)
            if strike <= 0:
                continue
                
            delta = abs(black_scholes_delta(spot, strike, T, r=0.045, sigma=sigma, option_type=opt_type))
            diff = abs(delta - 0.75)
            if diff < best_diff:
                best_diff = diff
                best_strike = strike
                
        return best_strike

    async def run_backtest(self, tickers: list[str], max_holding_days: int = 10, lookback_days: int = 365):
        """Runs the historical backtest simulating 14-21 DTE 0.75 Delta ITM options with exchange strikes and slippage."""
        log.info(f"Starting 14-21 DTE (18d model) 0.75 Delta ITM Options Backtest on {len(tickers)} tickers ({lookback_days} days lookback, max {max_holding_days}d hold)...")
        
        all_signals = []
        strategy = DonchianSwingStrategy(self.tradier_token)
        
        async with aiohttp.ClientSession() as session:
            for ticker in tickers:
                log.info(f"Backtesting {ticker}...")
                df = await self.fetch_historical_daily(session, ticker, days=lookback_days)
                if df.empty or len(df) < 60:
                    continue
                    
                # Compute strategy indicators
                df = strategy.compute_indicators(df)
                
                # Compute 20-day historical annualized volatility
                df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
                df["hist_vol_20"] = df["log_ret"].rolling(20).std() * math.sqrt(252)
                df["hist_vol_20"] = df["hist_vol_20"].fillna(0.35).clip(lower=0.15, upper=1.20)
                
                for i in range(60, len(df) - 1):
                    window_df = df.iloc[:i+1]
                    signals = strategy.evaluate_signals(window_df, ticker)
                    
                    if signals:
                        for sig in signals:
                            entry_spot = sig["entry_price"]
                            tp_spot = sig["take_profit"]
                            sl_spot = sig["stop_loss"]
                            direction = sig["direction"]
                            opt_type = "call" if direction == "Long" else "put"
                            
                            # Tiered Velocity Gate Check
                            tier_info = get_asset_tier_info(ticker)
                            expected_move_pct = (abs(tp_spot - entry_spot) / entry_spot) * 100.0 if entry_spot > 0 else 0.0
                            if expected_move_pct < tier_info["min_move_pct"] and sig.get("z_vol", 0.0) < 1.8:
                                continue # Filter out sluggish moves below velocity requirement
                            
                            sigma = float(window_df.iloc[-1].get("hist_vol_20", 0.35))
                            initial_dte = 18 # 14-21 DTE sweet spot model (18 days to expiry)
                            
                            # Find 0.75 Delta strike snapped to exchange grid
                            strike = self.find_075_delta_strike(entry_spot, direction, sigma, initial_dte)
                            theor_entry = black_scholes_price(entry_spot, strike, initial_dte / 365.0, 0.045, sigma, opt_type)
                            # Realistic 3% market maker spread friction on entry (pay the Ask)
                            entry_opt_price = round(theor_entry * 1.03, 2)
                            initial_delta = black_scholes_delta(entry_spot, strike, initial_dte / 365.0, 0.045, sigma, opt_type)
                            initial_theta = black_scholes_theta(entry_spot, strike, initial_dte / 365.0, 0.045, sigma, opt_type)
                            
                            # Option-level profit targets
                            option_tp_target = entry_opt_price * 1.40  # +40% target
                            option_sl_target = entry_opt_price * 0.70  # -30% stop
                            
                            outcome = "EXPIRED"
                            days_held = 0
                            sig["same_day_ambiguous"] = False
                            final_opt_price = entry_opt_price
                            max_opt_gain_pct = 0.0
                            
                            for j in range(i+1, min(i + max_holding_days + 1, len(df))):
                                days_held += 1
                                future_high = df.iloc[j]["high"]
                                future_low = df.iloc[j]["low"]
                                future_close = df.iloc[j]["close"]
                                rem_dte = max(0.5, initial_dte - days_held)
                                
                                # Best and worst intraday option prices on day j
                                best_spot = future_high if direction == "Long" else future_low
                                worst_spot = future_low if direction == "Long" else future_high
                                
                                # Deduct 3% market maker spread on exit (sell to the Bid)
                                best_opt_price = round(black_scholes_price(best_spot, strike, rem_dte / 365.0, 0.045, sigma, opt_type) * 0.97, 2)
                                worst_opt_price = round(black_scholes_price(worst_spot, strike, rem_dte / 365.0, 0.045, sigma, opt_type) * 0.97, 2)
                                close_opt_price = round(black_scholes_price(future_close, strike, rem_dte / 365.0, 0.045, sigma, opt_type) * 0.97, 2)
                                
                                gain_pct = (best_opt_price - entry_opt_price) / entry_opt_price
                                if gain_pct > max_opt_gain_pct:
                                    max_opt_gain_pct = gain_pct
                                    
                                hit_win = (best_opt_price >= option_tp_target) or (future_high >= tp_spot if direction == "Long" else future_low <= tp_spot)
                                hit_loss = (worst_opt_price <= option_sl_target) or (future_low <= sl_spot if direction == "Long" else future_high >= sl_spot)
                                
                                if hit_win and hit_loss:
                                    sig["same_day_ambiguous"] = True
                                    outcome = "LOSS" # Conservative tiebreak
                                    final_opt_price = option_sl_target
                                    break
                                elif hit_loss:
                                    outcome = "LOSS"
                                    final_opt_price = worst_opt_price
                                    break
                                elif hit_win:
                                    outcome = "WIN"
                                    final_opt_price = best_opt_price
                                    break
                                else:
                                    final_opt_price = close_opt_price
                                    
                            opt_pnl_pct = ((final_opt_price - entry_opt_price) / entry_opt_price) * 100.0
                            
                            sig["asset_tier"] = tier_info["tier"]
                            sig["expected_move_pct"] = round(expected_move_pct, 1)
                            sig["option_strike"] = strike
                            sig["option_entry_price"] = round(entry_opt_price, 2)
                            sig["option_final_price"] = round(final_opt_price, 2)
                            sig["option_pnl_pct"] = round(opt_pnl_pct, 1)
                            sig["max_gain_pct"] = round(max_opt_gain_pct * 100.0, 1)
                            sig["initial_delta"] = round(initial_delta, 2)
                            sig["initial_theta"] = round(initial_theta, 4)
                            sig["hist_vol_20"] = round(sigma, 4)
                            sig["outcome"] = outcome
                            sig["days_held"] = days_held
                            sig["backtest_date"] = window_df.index[-1].strftime("%Y-%m-%d")
                            all_signals.append(sig)
                            
        # Write to CSV
        if all_signals:
            filename = os.path.join(self.results_dir, f"backtest_itm_075delta_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
            keys = [
                "backtest_date", "ticker", "asset_tier", "expected_move_pct", "direction", "direction_code", "catalyst_type", "entry_price", "option_strike",
                "initial_delta", "initial_theta", "option_entry_price", "option_final_price", "option_pnl_pct",
                "max_gain_pct", "outcome", "days_held", "same_day_ambiguous", "z_vol", "rsi_14", "chop_14",
                "hist_vol_20", "sma20_ratio", "sma_spread", "breakout_pct"
            ]
            
            with open(filename, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys, extrasaction='ignore')
                dict_writer.writeheader()
                dict_writer.writerows(all_signals)
                
            log.info(f"✅ Backtest complete! Exported {len(all_signals)} trades to {filename}")
            
            # Print performance analytics
            wins = sum(1 for s in all_signals if s["outcome"] == "WIN")
            losses = sum(1 for s in all_signals if s["outcome"] == "LOSS")
            expired = sum(1 for s in all_signals if s["outcome"] == "EXPIRED")
            total = len(all_signals)
            
            total_wins_pnl = sum(s["option_pnl_pct"] for s in all_signals if s["outcome"] == "WIN")
            total_losses_pnl = abs(sum(s["option_pnl_pct"] for s in all_signals if s["outcome"] in ("LOSS", "EXPIRED")))
            profit_factor = (total_wins_pnl / total_losses_pnl) if total_losses_pnl > 0 else 99.0
            avg_pnl = sum(s["option_pnl_pct"] for s in all_signals) / total if total > 0 else 0.0
            ambiguous_count = sum(1 for s in all_signals if s.get("same_day_ambiguous"))
            
            print("\n" + "="*60)
            print("  --- 0.75 DELTA ITM OPTIONS BACKTEST SUMMARY ---")
            print("="*60)
            print(f"Total Signals Generated: {total}")
            print(f"Wins: {wins} | Losses: {losses} | Expired: {expired}")
            if (wins + losses) > 0:
                print(f"Win Rate (Wins / Decided): {(wins / (wins + losses))*100:.1f}%")
            print(f"Overall Strategy Win Rate: {(wins / total)*100:.1f}%")
            print(f"Average Option Return per Trade: {avg_pnl:+.1f}%")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Same-day Ambiguous Cases: {ambiguous_count} (conservatively counted as LOSS)")
            print("="*60 + "\n")
        else:
            log.info("No signals generated during backtest.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))
    
    logging.basicConfig(level=logging.INFO)
    token = os.getenv("TRADIER_ACCESS_TOKEN")
    
    engine = BacktestEngine(token)
    # High-volatility & Mega-cap basket
    basket = [
        "NVDA", "TSLA", "AAPL", "SPY", "QQQ", "AMD", "META", "AMZN", 
        "MSFT", "GOOGL", "AVGO", "PLTR", "ARM", "CRWD", "NFLX", "SHOP"
    ]
    asyncio.run(engine.run_backtest(basket, max_holding_days=10, lookback_days=365))

