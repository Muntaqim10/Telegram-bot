import os
import logging
import asyncio
import aiohttp
from datetime import date, timedelta, datetime
from typing import Optional

log = logging.getLogger(__name__)

class EarningsCalendar:
    """
    Fetches the next upcoming earnings date for a given ticker using the Finnhub API.
    Caches responses to avoid redundant API calls.
    """
    def __init__(self, finnhub_token: str = None):
        self.finnhub_token = finnhub_token or os.getenv("FINNHUB_TOKEN")
        self._cache = {}  # ticker -> {"next_earnings_date": Optional[str], "fetched_at": datetime}
        self.cache_ttl = 24 * 3600  # 24 hours in seconds

    async def get_next_earnings_date(self, ticker: str, session: aiohttp.ClientSession = None) -> Optional[str]:
        """
        Retrieves the next upcoming earnings date for a symbol.
        Returns a "YYYY-MM-DD" string or None.
        """
        now = datetime.now()
        
        # Check cache
        if ticker in self._cache:
            cached = self._cache[ticker]
            age = (now - cached["fetched_at"]).total_seconds()
            if age < self.cache_ttl:
                log.debug(f"Earnings cache hit for {ticker}. Age: {age:.1f}s")
                return cached["next_earnings_date"]

        next_date = None
        if self.finnhub_token:
            next_date = await self._fetch_finnhub(ticker, session)
            
        # Update cache (we cache even if None to prevent hammering the API for missing data)
        self._cache[ticker] = {"next_earnings_date": next_date, "fetched_at": now}

        return next_date

    async def _fetch_finnhub(self, ticker: str, session: aiohttp.ClientSession = None) -> Optional[str]:
        today = datetime.now()
        from_str = today.strftime("%Y-%m-%d")
        to_str = (today + timedelta(days=45)).strftime("%Y-%m-%d")
        
        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {
            "symbol": ticker,
            "from": from_str,
            "to": to_str,
            "token": self.finnhub_token
        }
        
        async def do_fetch(s):
            async with s.get(url, params=params, timeout=10.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    calendar = data.get("earningsCalendar", [])
                    
                    # Filter for dates today or later
                    future_dates = []
                    for item in calendar:
                        date_str = item.get("date")
                        if date_str and date_str >= from_str:
                            future_dates.append(date_str)
                            
                    if future_dates:
                        # Sort and return the earliest upcoming date
                        future_dates.sort()
                        return future_dates[0]
                else:
                    log.warning(f"Finnhub earnings API returned status {resp.status} for {ticker}")
            return None
            
        try:
            if session:
                return await do_fetch(session)
            else:
                async with aiohttp.ClientSession() as s:
                    return await do_fetch(s)
        except Exception as e:
            log.warning(f"Failed to fetch Finnhub earnings for {ticker}: {e}")
        return None

    def days_until_earnings(self, next_earnings_date: Optional[str]) -> Optional[int]:
        """
        Helper to return the integer days from today until the next earnings date.
        """
        if not next_earnings_date:
            return None
            
        try:
            target_date = datetime.strptime(next_earnings_date, "%Y-%m-%d").date()
            today = date.today()
            delta = target_date - today
            return delta.days
        except ValueError:
            return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
    config_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/.env"))
    load_dotenv(dotenv_path=root_env)
    load_dotenv(dotenv_path=config_env)

    logging.basicConfig(level=logging.DEBUG)
    
    async def test():
        calendar = EarningsCalendar()
        date_str = await calendar.get_next_earnings_date("NVDA")
        days = calendar.days_until_earnings(date_str)
        print(f"NVDA Next Earnings Date: {date_str}")
        print(f"Days Until Earnings: {days}")
        
    asyncio.run(test())
