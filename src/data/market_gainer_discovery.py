import os
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any

log = logging.getLogger(__name__)

from src.data.dynamic_scanner import CANDIDATE_POOL

# Expanded 250+ Liquid, Optionable US Equities Universe (Synchronized with CANDIDATE_POOL)
EXPANDED_UNIVERSE = CANDIDATE_POOL

class MarketGainerDiscovery:
    """
    Multi-API Market-Wide Surge Discovery Engine.
    Concurrently polls:
      1. Yahoo Finance API (top 100 day_gainers, most_actives, day_losers across all US exchanges)
      2. Tradier Bulk REST Quotes (expanded 350+ liquid candidate universe for volume surges)
    Deduplicates, verifies minimum price & volume, and outputs top active momentum symbols.
    """
    def __init__(self, tradier_token: str = None):
        self.tradier_token = tradier_token or os.getenv("TRADIER_ACCESS_TOKEN")
        self.tradier_headers = {
            "Authorization": f"Bearer {self.tradier_token}",
            "Accept": "application/json"
        }
        self.yahoo_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

    async def fetch_yahoo_screeners(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Queries Yahoo Finance predefined screeners for day_gainers, most_actives, and day_losers."""
        endpoints = ["day_gainers", "most_actives", "day_losers"]
        results: List[Dict[str, Any]] = []

        for scr in endpoints:
            url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&lang=en-US&region=US&scrIds={scr}&count=100"
            try:
                async with session.get(url, headers=self.yahoo_headers, timeout=8.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
                        for q in quotes:
                            sym = q.get("symbol", "").upper()
                            price = q.get("regularMarketPrice", 0.0) or 0.0
                            pct_change = q.get("regularMarketChangePercent", 0.0) or 0.0
                            vol = q.get("regularMarketVolume", 0) or 0

                            # Strict Quality Filter: Price >= $5, Volume >= 500k, US ticker format
                            if sym and price >= 5.0 and vol >= 500_000 and "^" not in sym and "." not in sym:
                                results.append({
                                    "symbol": sym,
                                    "price": float(price),
                                    "pct_change": float(pct_change),
                                    "volume": int(vol),
                                    "source": f"yahoo_{scr}"
                                })
            except Exception as e:
                log.warning(f"Yahoo Finance screener fetch failed for {scr}: {e}")

        return results

    async def fetch_tradier_bulk(self, session: aiohttp.ClientSession, symbols: List[str]) -> List[Dict[str, Any]]:
        """Queries Tradier bulk quotes for volume surges across the expanded universe."""
        if not self.tradier_token or not symbols:
            return []

        results: List[Dict[str, Any]] = []
        # Tradier accepts up to 500 symbols in a single call
        symbols_str = ",".join(symbols[:450])
        url = f"https://api.tradier.com/v1/markets/quotes?symbols={symbols_str}&greeks=false"

        try:
            async with session.get(url, headers=self.tradier_headers, timeout=8.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    quotes = data.get("quotes", {}).get("quote", [])
                    if not isinstance(quotes, list):
                        quotes = [quotes]

                    for q in quotes:
                        sym = q.get("symbol", "").upper()
                        last_price = q.get("last") or q.get("close") or 0.0
                        vol = q.get("volume", 0) or 0
                        avg_vol = q.get("average_volume", 0) or 1
                        pct_change = q.get("change_percentage", 0.0) or 0.0

                        if sym and last_price >= 5.0 and vol >= 200_000:
                            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
                            results.append({
                                "symbol": sym,
                                "price": float(last_price),
                                "pct_change": float(pct_change),
                                "volume": int(vol),
                                "vol_ratio": float(vol_ratio),
                                "source": "tradier_bulk"
                            })
        except Exception as e:
            log.warning(f"Tradier bulk quotes fetch failed: {e}")

        return results

    async def discover_market_movers(self, max_symbols: int = 150) -> List[str]:
        """
        Executes dual-API discovery across Tradier & Yahoo Finance in parallel.
        Merges, scores, deduplicates, and returns the top active momentum tickers.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Concurrent execution of both Yahoo Finance and Tradier
                yahoo_task = self.fetch_yahoo_screeners(session)
                tradier_task = self.fetch_tradier_bulk(session, EXPANDED_UNIVERSE)

                yahoo_movers, tradier_movers = await asyncio.gather(yahoo_task, tradier_task, return_exceptions=True)

                if isinstance(yahoo_movers, Exception):
                    log.warning(f"Yahoo movers exception: {yahoo_movers}")
                    yahoo_movers = []
                if isinstance(tradier_movers, Exception):
                    log.warning(f"Tradier movers exception: {tradier_movers}")
                    tradier_movers = []

                # Combine & Score Candidates
                scored_pool: Dict[str, float] = {}

                # 1. Score Tradier Universe Movers
                for tm in tradier_movers:
                    sym = tm["symbol"]
                    vol_ratio = tm.get("vol_ratio", 1.0)
                    pct_change = abs(tm.get("pct_change", 0.0))
                    # Score = volume surge weight + price expansion
                    score = (vol_ratio * 3.0) + (pct_change * 1.5)
                    # Priority for major core index/tech leaders
                    if sym in ["SPY", "QQQ", "IWM", "GLD", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"]:
                        score += 500.0
                    scored_pool[sym] = max(scored_pool.get(sym, 0.0), score)

                # 2. Score Yahoo Finance Market-Wide Gainers/Movers (Filtered strictly to backtested universe)
                candidate_set = set(EXPANDED_UNIVERSE)
                for ym in yahoo_movers:
                    sym = ym["symbol"]
                    if sym not in candidate_set:
                        continue
                    pct_change = abs(ym.get("pct_change", 0.0))
                    vol = ym.get("volume", 0)
                    dollar_vol = vol * ym.get("price", 10.0)
                    # Score = high % change + dollar liquidity
                    score = (pct_change * 2.5) + (dollar_vol / 2e8)
                    scored_pool[sym] = max(scored_pool.get(sym, 0.0), score)

                # Always guarantee core indices & Mag 7
                core_anchors = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"]
                for core in core_anchors:
                    if core not in scored_pool:
                        scored_pool[core] = 1000.0

                # Sort by score descending
                sorted_tickers = sorted(scored_pool.items(), key=lambda x: x[1], reverse=True)
                top_symbols = [t[0] for t in sorted_tickers[:max_symbols]]

                log.info(
                    f"🚀 [MULTI-API DISCOVERY] Found {len(yahoo_movers)} Yahoo movers + {len(tradier_movers)} Tradier quotes. "
                    f"Consolidated top {len(top_symbols)} active surge tickers."
                )
                return top_symbols

        except Exception as e:
            log.error(f"Failed in discover_market_movers: {e}")
            return EXPANDED_UNIVERSE[:max_symbols]
