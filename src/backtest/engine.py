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
    Downloads historical data, runs the strategy, applies FlashAlpha GEX filters,
    and outputs the result to a CSV file.
    """
    def __init__(self, tradier_token: str, flashalpha_key: str):
        self.tradier_token = tradier_token
        self.flashalpha_key = flashalpha_key
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
                            entry_price = sig["entry_price"]
                            tp = sig["take_profit"]
                            sl = sig["stop_loss"]
                            direction = sig["direction"]
                            
                            outcome = "PENDING"
                            days_held = 0
                            
                            # Check next 20 days to see if it hit TP or SL
                            for j in range(i+1, min(i+21, len(df))):
                                days_held += 1
                                future_high = df.iloc[j]["high"]
                                future_low = df.iloc[j]["low"]
                                
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
            keys = ["backtest_date", "ticker", "direction", "catalyst_type", "entry_price", "stop_loss", "take_profit", "outcome", "days_held", "z_vol", "rsi_14", "chop_14"]
            
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
        else:
            log.info("No signals generated during backtest.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env")))
    
    logging.basicConfig(level=logging.INFO)
    
    token = os.getenv("TRADIER_ACCESS_TOKEN")
    fa_key = os.getenv("FLASHALPHA_API_KEY")
    
    engine = BacktestEngine(token, fa_key)
    # Test on a small basket
    asyncio.run(engine.run_backtest(["AAPL", "TSLA", "NVDA", "SPY", "QQQ"]))
