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

TIINGO_TOKEN = os.getenv("TIINGO_TOKEN")

# Major Indexes & Mag 7 Stocks
MAG7_AND_INDEXES = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"
]

# Use the actual universe the bot trades for ML training
from src.data.dynamic_scanner import CANDIDATE_POOL
VOLATILE_TICKERS = CANDIDATE_POOL

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/intraday/"))
os.makedirs(DATA_DIR, exist_ok=True)

async def fetch_1m_data(session: aiohttp.ClientSession, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches 1-minute historical data from Tiingo's IEX intraday endpoint.
    Provides deep historical data (up to several years) for robust ML training.
    """
    url = f"https://api.tiingo.com/iex/{ticker}/prices"
    headers = {
        "Authorization": f"Token {TIINGO_TOKEN}",
        "Content-Type": "application/json"
    }
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "resampleFreq": "1min",
        "columns": "open,high,low,close,volume"
    }
    
    for attempt in range(1, 4):
        try:
            async with session.get(url, headers=headers, params=params, timeout=20.0) as resp:
                if resp.status != 200:
                    log.warning(f"Failed to fetch data for {ticker}: status {resp.status}")
                    return pd.DataFrame()
                    
                data = await resp.json()
                if not data:
                    log.warning(f"No 1-minute data returned for {ticker}.")
                    return pd.DataFrame()
                    
                df = pd.DataFrame(data)
                if df.empty:
                    return df
                
                # Tiingo format: [{'date': '2023-01-03T14:30:00+00:00', 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}]
                df.rename(columns={'date': 'time'}, inplace=True)
                df["time"] = pd.to_datetime(df["time"])
                
                # Convert to US/Eastern timezone to match live execution logic expectations
                if df["time"].dt.tz is not None:
                    df["time"] = df["time"].dt.tz_convert('US/Eastern').dt.tz_localize(None)
                    
                df.set_index("time", inplace=True)
                
                # Filter strictly to regular market hours (9:30 AM to 4:00 PM EST)
                df = df.between_time("09:30", "16:00")
                
                return df
        except Exception as e:
            log.warning(f"Attempt {attempt}/3 failed to fetch 1m data for {ticker}: {e}")
            if attempt < 3:
                await asyncio.sleep(2)
    return pd.DataFrame()

async def main():
    if not TIINGO_TOKEN:
        log.error("Missing TIINGO_API_KEY in environment.")
        return
        
    # Tiingo allows deep historical fetch, we will grab 180 days (6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    log.info(f"Downloading 1-minute intraday data from {start_str} to {end_str} for {len(VOLATILE_TICKERS)} tickers...")
    
    async with aiohttp.ClientSession() as session:
        for ticker in VOLATILE_TICKERS:
            file_path = os.path.join(DATA_DIR, f"{ticker}_1m.csv")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 10000:
                log.info(f"Using cached intraday CSV for {ticker}")
                continue
                
            log.info(f"Fetching {ticker} (180 days)...")
            df = await fetch_1m_data(session, ticker, start_str, end_str)
            if not df.empty:
                df.to_csv(file_path)
                log.info(f"Saved {len(df)} rows for {ticker} to {file_path}")
            else:
                log.warning(f"No data saved for {ticker}.")
            
            # Rate limiting safety for Tiingo
            await asyncio.sleep(1.0)
            
    log.info("Historical download complete.")

if __name__ == "__main__":
    asyncio.run(main())
