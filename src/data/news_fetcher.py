import os
import logging
import asyncio
import aiohttp
from datetime import date, timedelta, datetime
from typing import List

log = logging.getLogger(__name__)

class NewsFetcher:
    """
    Fetches real-time financial news headlines for tickers using Finnhub API with Tiingo fallback.
    """
    def __init__(self, finnhub_token: str = None, tiingo_token: str = None):
        self.finnhub_token = finnhub_token or os.getenv("FINNHUB_TOKEN")
        self.tiingo_token = tiingo_token or os.getenv("TIINGO_TOKEN")
        self._cache = {}  # (ticker, date) -> {"headlines": List[str], "fetched_at": datetime}
        self.cache_ttl = int(os.getenv("NEWS_CACHE_TTL_SECONDS", "300"))

    async def get_headlines(self, ticker: str, max_count: int = 10) -> List[str]:
        """
        Retrieves recent headlines for a symbol with in-memory caching.
        """
        today = date.today()
        cache_key = (ticker, today)
        now = datetime.now()

        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            age = (now - cached["fetched_at"]).total_seconds()
            if age < self.cache_ttl:
                log.debug(f"News cache hit for {ticker}. Age: {age:.1f}s")
                return cached["headlines"][:max_count]

        headlines = []
        if self.finnhub_token:
            headlines = await self._fetch_finnhub(ticker, max_count)
            
        if not headlines and self.tiingo_token:
            headlines = await self._fetch_tiingo(ticker, max_count)

        # Update cache
        if headlines:
            self._cache[cache_key] = {"headlines": headlines, "fetched_at": now}

        return headlines

    async def _fetch_finnhub(self, ticker: str, max_count: int) -> List[str]:
        today_str = date.today().strftime("%Y-%m-%d")
        yesterday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={yesterday_str}&to={today_str}&token={self.finnhub_token}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        headlines = [item.get("headline", "").strip() for item in data if item.get("headline")]
                        return headlines[:max_count]
                    else:
                        log.warning(f"Finnhub news API returned status {resp.status} for {ticker}")
        except Exception as e:
            log.warning(f"Failed to fetch Finnhub news for {ticker}: {e}")
        return []

    async def _fetch_tiingo(self, ticker: str, max_count: int) -> List[str]:
        url = f"https://api.tiingo.com/tiingo/news?tickers={ticker}&token={self.tiingo_token}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        headlines = [item.get("title", "").strip() for item in data if item.get("title")]
                        return headlines[:max_count]
        except Exception as e:
            log.warning(f"Failed to fetch Tiingo news for {ticker}: {e}")
        return []
