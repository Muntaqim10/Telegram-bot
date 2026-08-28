import os
import asyncio
import aiohttp
import pandas as pd
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
config_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/.env"))
load_dotenv(dotenv_path=root_env)
load_dotenv(dotenv_path=config_env)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("historical_downloader")

TIINGO_TOKEN = os.getenv("TIINGO_TOKEN")

# Major Indexes & Mag 7 Stocks
# Major Indexes & Mag 7 Stocks (Tiingo free tier rate-limit safe: 11 core benchmark tickers)
MAG7_AND_INDEXES = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"
]
VOLATILE_TICKERS = MAG7_AND_INDEXES

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/intraday/"))
os.makedirs(DATA_DIR, exist_ok=True)

async def fetch_1m_data(session: aiohttp.ClientSession, ticker: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, bool]:
    """
    Fetches 1-minute historical data from Tiingo's IEX intraday endpoint.
    Provides historical benchmark data for index/Mag 7 correlation.
    Returns (DataFrame, is_rate_limited).
    """
    url = f"https://api.tiingo.com/iex/{ticker}/prices"
    headers = {
        "Authorization": f"Token {TIINGO_TOKEN}",
        "Content-Type": "application/json"
    }
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    all_dfs = []
    current_start = start_dt
    is_rate_limited = False
    
    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=20), end_dt)
        
        params = {
            "startDate": current_start.strftime("%Y-%m-%d"),
            "endDate": current_end.strftime("%Y-%m-%d"),
            "resampleFreq": "1min",
            "columns": "open,high,low,close,volume"
        }
        
        for attempt in range(1, 4):
            try:
                async with session.get(url, headers=headers, params=params, timeout=20.0) as resp:
                    if resp.status == 429:
                        log.warning(f"Tiingo API hourly rate limit (429) hit for {ticker}. Halting download to avoid quota burn.")
                        is_rate_limited = True
                        break
                    elif resp.status != 200:
                        log.warning(f"Failed to fetch chunk for {ticker} ({current_start.date()}): status {resp.status}")
                        break
                        
                    data = await resp.json()
                    if data:
                        df = pd.DataFrame(data)
                        if not df.empty:
                            df.rename(columns={'date': 'time'}, inplace=True)
                            df["time"] = pd.to_datetime(df["time"])
                            if df["time"].dt.tz is not None:
                                df["time"] = df["time"].dt.tz_convert('US/Eastern').dt.tz_localize(None)
                            df.set_index("time", inplace=True)
                            df = df.between_time("09:30", "16:00")
                            all_dfs.append(df)
                    

                    break
            except Exception as e:
                log.warning(f"Attempt {attempt}/3 failed to fetch 1m data for {ticker}: {e}")
                if attempt < 3:
                    await asyncio.sleep(2)
                    
        if is_rate_limited:
            break

        # Advance by 21 days so we don't overlap the end date of the previous chunk
        current_start = current_end + timedelta(days=1)
        await asyncio.sleep(0.5) # Rate limit between chunks
        
    final_df = pd.DataFrame()
    if all_dfs:
        final_df = pd.concat(all_dfs)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df.sort_index(inplace=True)
        
    return final_df, is_rate_limited

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
            df, is_rate_limited = await fetch_1m_data(session, ticker, start_str, end_str)
            if not df.empty:
                df.to_csv(file_path)
                log.info(f"Saved {len(df)} rows for {ticker} to {file_path}")
            else:
                log.warning(f"No data saved for {ticker}.")

            if is_rate_limited:
                log.warning("Tiingo quota/hourly rate limit (429) hit. Stopping download to prevent API lockout.")
                break
            
            # Rate limiting safety for Tiingo
            await asyncio.sleep(1.0)
            
    log.info("Historical download complete.")

if __name__ == "__main__":
    asyncio.run(main())
