import os
import logging
import asyncio
import aiohttp
from datetime import date, timedelta, datetime
from typing import Optional

log = logging.getLogger(__name__)


def market_today_date() -> date:
    """Today in market time, as a date.

    Every date in this system is US/Eastern so that entry dates, hold counts and
    expiration checks cannot disagree on a UTC host -- see RiskManager.market_today().
    date.today() is host time, which on a UTC box rolls over five hours early and makes
    an evening signal believe a print is already behind it.
    """
    import pandas as pd
    return pd.Timestamp.now(tz="US/Eastern").date()

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
        cache_key = (ticker, "next")
        
        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            age = (now - cached["fetched_at"]).total_seconds()
            if age < self.cache_ttl:
                log.debug(f"Earnings cache hit for {ticker} (next). Age: {age:.1f}s")
                return cached["date"]

        next_date = None
        if self.finnhub_token:
            next_date = await self._fetch_finnhub(ticker, "next", session)
            
        # Update cache (we cache even if None to prevent hammering the API for missing data)
        self._cache[cache_key] = {"date": next_date, "fetched_at": now}

        return next_date

    async def get_last_earnings_date(self, ticker: str, session: aiohttp.ClientSession = None) -> Optional[str]:
        """
        Retrieves the most recent past earnings date for a symbol.
        Returns a "YYYY-MM-DD" string or None.
        """
        now = datetime.now()
        cache_key = (ticker, "last")
        
        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            age = (now - cached["fetched_at"]).total_seconds()
            if age < self.cache_ttl:
                log.debug(f"Earnings cache hit for {ticker} (last). Age: {age:.1f}s")
                return cached["date"]

        last_date = None
        if self.finnhub_token:
            last_date = await self._fetch_finnhub(ticker, "last", session)
            
        self._cache[cache_key] = {"date": last_date, "fetched_at": now}

        return last_date

    async def _fetch_finnhub(self, ticker: str, direction: str, session: aiohttp.ClientSession = None) -> Optional[str]:
        today = datetime.now()
        if direction == "next":
            from_str = today.strftime("%Y-%m-%d")
            to_str = (today + timedelta(days=45)).strftime("%Y-%m-%d")
        else:
            from_str = (today - timedelta(days=45)).strftime("%Y-%m-%d")
            to_str = today.strftime("%Y-%m-%d")
        
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
                    
                    # Filter for dates based on direction
                    target_dates = []
                    for item in calendar:
                        date_str = item.get("date")
                        if not date_str:
                            continue
                        if direction == "next" and date_str >= from_str:
                            target_dates.append(date_str)
                        elif direction == "last" and date_str <= to_str:
                            target_dates.append(date_str)
                            
                    if target_dates:
                        # Sort and return the nearest date
                        target_dates.sort(reverse=(direction == "last"))
                        return target_dates[0]
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
            delta = target_date - market_today_date()
            return delta.days
        except ValueError:
            return None

    def days_since_earnings(self, last_earnings_date: Optional[str]) -> Optional[int]:
        """
        Helper to return the integer days from the last earnings date until today.
        """
        if not last_earnings_date:
            return None

        try:
            target_date = datetime.strptime(last_earnings_date, "%Y-%m-%d").date()
            delta = market_today_date() - target_date
            return delta.days
        except ValueError:
            return None

    @staticmethod
    def earnings_in_window(next_earnings: Optional[str], expiration: Optional[str],
                           today: Optional[str] = None) -> bool:
        """Does a known earnings date fall inside the contract's life?

        Fails open. The calendar does not know every ticker -- AAPL returns no date at
        all -- so an absent or unparseable date is "no known print", not "safe". Blocking
        every unknown would silence most alerts.

        Lives here rather than in the pipeline so the rule has one definition: the
        callers were a gate and a test asserting against its own copy of the comparison.
        """
        if not next_earnings or not expiration:
            return False
        try:
            earnings_dt = datetime.strptime(next_earnings, "%Y-%m-%d").date()
            expiration_dt = datetime.strptime(expiration, "%Y-%m-%d").date()
            today_dt = (datetime.strptime(today, "%Y-%m-%d").date() if today
                        else market_today_date())
        except (ValueError, TypeError):
            return False
        return today_dt <= earnings_dt <= expiration_dt

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
