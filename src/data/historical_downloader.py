import os
import time
import json
import asyncio
import aiohttp
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
config_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/.env"))
load_dotenv(dotenv_path=root_env)
load_dotenv(dotenv_path=config_env)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("historical_downloader")

TRADIER_TOKEN = os.getenv("TRADIER_ACCESS_TOKEN")

# Major Indexes & Mag 7 Stocks for institutional backtesting & ML training
MAG7_AND_INDEXES = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"
]
VOLATILE_TICKERS = MAG7_AND_INDEXES

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/intraday/"))
os.makedirs(DATA_DIR, exist_ok=True)

async def fetch_1m_data(session: aiohttp.ClientSession, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches 1-minute historical data from Tradier's timesales endpoint.
    Note: Tradier restricts 1-minute data to approximately the last 30-35 days.
    """
    url = "https://api.tradier.com/v1/markets/timesales"
    headers = {
        "Authorization": f"Bearer {TRADIER_TOKEN}",
        "Accept": "application/json"
    }
    params = {
        "symbol": ticker,
        "interval": "1min",
        "start": f"{start_date} 09:30",
        "end": f"{end_date} 16:00"
    }
    
    for attempt in range(1, 4):
        try:
            async with session.get(url, headers=headers, params=params, timeout=10.0) as resp:
                if resp.status != 200:
                    log.warning(f"Failed to fetch data for {ticker}: status {resp.status}")
                    return pd.DataFrame()
                    
                data = await resp.json()
                series = data.get("series", {})
                if not series or "data" not in series:
                    log.warning(f"No 1-minute data returned for {ticker}.")
                    return pd.DataFrame()
                    
                candles = series["data"]
                df = pd.DataFrame(candles)
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
                return df
        except Exception as e:
            log.warning(f"Attempt {attempt}/3 failed to fetch 1m data for {ticker}: {e}")
            if attempt < 3:
                await asyncio.sleep(2)
    return pd.DataFrame()

async def main():
    if not TRADIER_TOKEN:
        log.error("Missing TRADIER_ACCESS_TOKEN in environment.")
        return
        
    # Calculate the safest 25-day window for Tradier 1-minute timesales
    end_date = datetime.now()
    start_date = end_date - timedelta(days=25)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    log.info(f"Downloading 1-minute intraday data from {start_str} to {end_str} for {len(VOLATILE_TICKERS)} tickers...")
    
    async with aiohttp.ClientSession() as session:
        for ticker in VOLATILE_TICKERS:
            file_path = os.path.join(DATA_DIR, f"{ticker}_1m.csv")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 10000:
                log.info(f"Using cached intraday CSV for {ticker}")
                continue
                
            log.info(f"Fetching {ticker}...")
            df = await fetch_1m_data(session, ticker, start_str, end_str)
            if not df.empty:
                df.to_csv(file_path)
                log.info(f"Saved {len(df)} rows for {ticker} to {file_path}")
            
            # Rate limiting safety
            await asyncio.sleep(1.5)
            
    log.info("Historical download complete.")

if __name__ == "__main__":
    asyncio.run(main())
