import os
import logging
import asyncio
import aiohttp
from typing import List

log = logging.getLogger(__name__)

# Broad pool of dynamic candidates optimized for HIGH OPTIONS LEVERAGE (High Beta, High IV, Massive Gamma potential)
CANDIDATE_POOL = [
    # Core Indexes & Mag 7
    "SPY", "QQQ", "IWM", "GLD", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # High-Beta Semiconductors & Hardware
    "AMD", "AVGO", "SMCI", "ARM", "MU", "LRCX", "AMAT", "DELL", "ALAB", "INTC", "TXN",
    # Crypto Proxies & Ultra-High Beta
    "COIN", "MSTR", "MARA", "RIOT", "HOOD", "CVNA", "UPST", "AFRM", "SOUN",
    # High-Movement Software/Cloud/Cyber
    "PLTR", "CRWD", "PANW", "SNOW", "DDOG", "NET", "ZS", "ADBE", "INTU", "NOW",
    # High-Volatility Consumer/Retail/Tech
    "NFLX", "SHOP", "SQ", "ROKU", "UBER", "ABNB", "DASH", "DKNG", "CELH", "U", "RDDT",
    # Biotech/Pharma Runners
    "MRNA", "VRTX", "REGN", "LLY", "ISRG",
    # Meme & Retail Favorites
    "GME", "AMC", "DJT", "RIVN"
]

class DynamicTickerScanner:
    """
    Scans the broad market in real-time via Tradier API to identify tickers 
    currently experiencing high volume, volatility, and momentum surges.
    Eliminates static watchlists.
    """
    def __init__(self, tradier_token: str = None):
        self.token = tradier_token or os.getenv("TRADIER_ACCESS_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }

    async def get_active_market_movers(self, max_symbols: int = 50) -> List[str]:
        """
        Queries Tradier bulk quotes to filter for active volume surges and gappers.
        """
        if not self.token:
            log.warning("No Tradier token found. Falling back to default core symbols.")
            return CANDIDATE_POOL[:max_symbols]

        symbols_str = ",".join(CANDIDATE_POOL)
        url = f"https://api.tradier.com/v1/markets/quotes?symbols={symbols_str}&greeks=false"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=10.0) as resp:
                    if resp.status != 200:
                        log.warning(f"Tradier bulk quotes scanner status: {resp.status}")
                        return CANDIDATE_POOL[:max_symbols]

                    data = await resp.json()
                    quotes = data.get("quotes", {}).get("quote", [])
                    if not isinstance(quotes, list):
                        quotes = [quotes]

                    scored_tickers = []
                    for q in quotes:
                        sym = q.get("symbol")
                        vol = q.get("volume", 0) or 0
                        avg_vol = q.get("average_volume", 0) or 1
                        last_price = q.get("last") or q.get("close") or 0.0

                        if not sym or last_price <= 1.0:
                            continue

                        # Calculate Volume Surge & Dollar Volume score
                        vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
                        dollar_vol = vol * last_price

                        # Always include core major indexes & Mag 7
                        is_core = sym in ["SPY", "QQQ", "IWM", "GLD", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"]
                        
                        score = (vol_ratio * 2.0) + (dollar_vol / 1e8)
                        if is_core:
                            score += 1000.0 # Force priority for major leaders

                        scored_tickers.append((score, sym))

                    scored_tickers.sort(key=lambda x: x[0], reverse=True)
                    top_tickers = [t[1] for t in scored_tickers[:max_symbols]]
                    
                    log.info(f"⚡ [DYNAMIC SCANNER] Discovered Top {len(top_tickers)} active volume & momentum leaders from Tradier.")
                    return top_tickers

        except Exception as e:
            log.error(f"Error running DynamicTickerScanner: {e}")
            return CANDIDATE_POOL[:max_symbols]
