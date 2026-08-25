import os
import csv
import sys
import logging
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.strategy.donchian_daily import DonchianSwingStrategy

log = logging.getLogger(__name__)

class BacktestEngine:
    """
    Headless Backtesting Engine.
    Downloads historical daily bars from Tradier, runs the Donchian 
    swing strategy, and outputs signal outcomes to a CSV file.
    
    Note: does NOT currently apply FlashAlpha GEX filtering -- 
    historical GEX data requires a premium FlashAlpha plan. This 
    engine tests the base strategy only. See src/backtester_v2.py 
    for the primary backtester used to generate ML training labels; 
    this is a separate, standalone tool and the two are not 
    guaranteed to produce matching win rates (see the tiebreak 
    comment in run_backtest for one known methodological difference).
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
        
        async with session.get(url, headers=headers, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                days_data = data.get("history", {}).get("day", [])
                if not isinstance(days_data, list):
                    days_data = [days_data]
                df = pd.DataFrame(days_data)
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df.dropna(subset=["close"])
        return pd.DataFrame()

    async def run_backtest(self, tickers: list[str]):
        """Runs the historical backtest and saves to CSV."""
        log.info(f"Starting Backtest on {len(tickers)} tickers...")
        
        all_signals = []
        strategy = DonchianSwingStrategy(self.tradier_token)
        
        async with aiohttp.ClientSession() as session:
            for ticker in tickers:
                log.info(f"Backtesting {ticker}...")
                df = await self.fetch_historical_daily(session, ticker, days=365)
                if df.empty:
                    continue
                    
                # Compute strategy indicators
                df = strategy.compute_indicators(df)
                
                # Step through history (simulate walk-forward)
                # In a real backtest, we would also query FlashAlpha historical GEX here.
                # Since historical GEX requires premium FlashAlpha data, we will run the base 
                # Donchian strategy to generate signals for the CSV.
                
                for i in range(60, len(df)):
                    # Provide data up to index i
                    window_df = df.iloc[:i+1]
                    signals = strategy.evaluate_signals(window_df, ticker)
                    
                    if signals:
                        for sig in signals:
                            # Forward-looking calculation (Did it hit TP or SL?)
                            tp = sig["take_profit"]
                            sl = sig["stop_loss"]
                            direction = sig["direction"]
                            
                            outcome = "PENDING"
                            days_held = 0
                            sig["same_day_ambiguous"] = False
                            
                            # Check next 20 days to see if it hit TP or SL
                            for j in range(i+1, min(i+21, len(df))):
                                days_held += 1
                                future_high = df.iloc[j]["high"]
                                future_low = df.iloc[j]["low"]
                                
                                # ASSUMPTION: when a single day's high/low range touches BOTH 
                                # stop-loss and take-profit, we cannot know from daily OHLC data 
                                # which was hit first intraday. We conservatively assume 
                                # stop-loss hits first (worst-case ordering). This differs from 
                                # backtester_v2.py, which checks a single closing price rather 
                                # than the day's full range -- the two engines are NOT expected 
                                # to produce identical win rates on the same data, and that is 
                                # expected, not a bug.
                                if direction == "Long":
                                    if future_low <= sl and future_high >= tp:
                                        sig["same_day_ambiguous"] = True
                                else:
                                    if future_high >= sl and future_low <= tp:
                                        sig["same_day_ambiguous"] = True
                                
                                if direction == "Long":
                                    if future_low <= sl:
                                        outcome = "LOSS"
                                        break
                                    elif future_high >= tp:
                                        outcome = "WIN"
                                        break
                                else:
                                    if future_high >= sl:
                                        outcome = "LOSS"
                                        break
                                    elif future_low <= tp:
                                        outcome = "WIN"
                                        break
                                        
                            sig["outcome"] = outcome
                            sig["days_held"] = days_held
                            sig["backtest_date"] = window_df.index[-1].strftime("%Y-%m-%d")
                            all_signals.append(sig)
                            
        # Write to CSV
        if all_signals:
            filename = os.path.join(self.results_dir, f"backtest_donchian_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
            keys = ["backtest_date", "ticker", "direction", "catalyst_type", "entry_price", "stop_loss", "take_profit", "outcome", "days_held", "same_day_ambiguous", "z_vol", "rsi_14", "chop_14"]
            
            with open(filename, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys, extrasaction='ignore')
                dict_writer.writeheader()
                dict_writer.writerows(all_signals)
                
            log.info(f"✅ Backtest complete! Exported {len(all_signals)} signals to {filename}")
            
            # Print quick stats
            wins = sum(1 for s in all_signals if s["outcome"] == "WIN")
            losses = sum(1 for s in all_signals if s["outcome"] == "LOSS")
            total = wins + losses
            if total > 0:
                win_rate = (wins / total) * 100
                log.info(f"📊 Strategy Win Rate: {win_rate:.1f}% ({wins} W / {losses} L)")
                ambiguous_count = sum(1 for s in all_signals if s.get("same_day_ambiguous"))
                log.info(f"⚠️ {ambiguous_count} of {total} trades had same-day TP/SL ambiguity (resolved as LOSS)")
        else:
            log.info("No signals generated during backtest.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env")))
    
    logging.basicConfig(level=logging.INFO)
    
    token = os.getenv("TRADIER_ACCESS_TOKEN")
    
    engine = BacktestEngine(token)
    # Test on a small basket
    asyncio.run(engine.run_backtest(["AAPL", "TSLA", "NVDA", "SPY", "QQQ"]))
