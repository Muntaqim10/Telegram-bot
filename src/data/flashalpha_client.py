import os
import aiohttp
import logging
import asyncio
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

class FlashAlphaClient:
    """
    Client for interacting with the FlashAlpha Options Analytics API.
    Provides institutional-grade Gamma Exposure (GEX), Call Walls, and Put Walls.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FLASHALPHA_API_KEY")
        self.base_url = "https://lab.flashalpha.com/v1"
        self.headers = {
            "X-Api-Key": self.api_key or "",
            "Accept": "application/json"
        }
        self._cache = {}  # (ticker, expiration) -> (timestamp, result)
        
    async def get_gex_profile(self, ticker: str, expiration: str, session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
        """
        Fetches the GEX profile for a specific expiration date.
        Returns total Net GEX, Call Wall, Put Wall, and Gamma Flip Point.
        """
        if not self.api_key:
            log.warning("FlashAlpha API key not found. Skipping GEX lookup.")
            return {"status": "unavailable"}

        cache_key = (ticker.upper(), expiration)
        import time
        now = time.time()
        if cache_key in self._cache:
            ts, res = self._cache[cache_key]
            if now - ts < 300.0:  # 5-minute cache
                return res

        url = f"{self.base_url}/exposure/gex/{ticker.upper()}?expiration={expiration}"
        
        async def do_fetch(s):
            async with s.get(url, headers=self.headers, timeout=5.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "error" in data:
                        log.warning(f"FlashAlpha returned error for {ticker}: {data.get('message')}")
                        msg = data.get("message", "").lower()
                        if "quota" in msg or "rate" in msg:
                            return {"status": "rate_limited"}
                        return {"status": "unavailable"}
                    out = {
                        "status": "ok",
                        "net_gex": data.get("net_gex", 0.0),
                        "call_wall": data.get("call_wall", 0.0),
                        "put_wall": data.get("put_wall", 0.0),
                        "gamma_flip": data.get("gamma_flip_point", 0.0),
                        "spot_price": data.get("spot_price", 0.0)
                    }
                    self._cache[cache_key] = (now, out)
                    return out
                elif resp.status == 429:
                    log.warning(f"FlashAlpha API rate limit hit for {ticker}")
                    return {"status": "rate_limited"}
                else:
                    log.warning(f"FlashAlpha API returned status {resp.status} for {ticker}")
                    return {"status": "unavailable"}

        try:
            if session:
                return await do_fetch(session)
            else:
                async with aiohttp.ClientSession() as s:
                    return await do_fetch(s)
        except asyncio.TimeoutError:
            log.warning(f"FlashAlpha API timed out for {ticker}")
            return {"status": "unavailable"}
        except Exception as e:
            log.error(f"FlashAlpha API error for {ticker}: {e}")
            return {"status": "unavailable"}
