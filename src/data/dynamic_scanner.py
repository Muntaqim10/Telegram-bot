import os
import logging
import aiohttp
from typing import List

log = logging.getLogger(__name__)

# Broad pool of dynamic candidates optimized for HIGH OPTIONS LEVERAGE (High Beta, High IV, Massive Gamma potential)
CANDIDATE_POOL = [
    # Core Indexes & Mag 7
    "SPY", "QQQ", "IWM", "DIA", "GLD", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # High-Beta Semiconductors & AI Hardware
    "AMD", "AVGO", "TSM", "QCOM", "MRVL", "ASML", "SMCI", "ARM", "MU", "LRCX", "AMAT", "DELL", "ALAB", "INTC", "TXN",
    # Nuclear Energy & AI Power Infrastructure
    "CEG", "VST", "CCJ", "SMR", "OKLO", "FSLR", "ENPH",
    # Defense, Aerospace & Space Tech
    "LMT", "RTX", "NOC", "BA", "RKLB", "ASTS", "ACHR", "JOBY",
    # Quantum & Next-Gen Computing
    "IONQ", "RGTI", "QBTS",
    # Crypto Proxies & High-Beta Fintech
    "COIN", "MSTR", "MARA", "RIOT", "CLSK", "WULF", "CIFR", "HOOD", "CVNA", "UPST", "AFRM", "SOFI", "SOUN", "APP", "PYPL",
    # Cloud, Cybersecurity & Enterprise SaaS
    "PLTR", "CRWD", "PANW", "SNOW", "DDOG", "NET", "ZS", "ADBE", "INTU", "NOW", "FTNT", "MDB", "TEAM", "PATH",
    # High-Growth Platforms, Consumer, Retail & Media
    "NFLX", "DIS", "SHOP", "SQ", "ROKU", "UBER", "ABNB", "DASH", "DKNG", "CELH", "U", "RDDT", "CAVA", "HIMS", "TEM", "LULU", "NKE", "COST", "WMT", "TGT",
    # Biotech & Healthcare Movers
    "MRNA", "VRTX", "REGN", "LLY", "ISRG", "NVO", "BIIB", "GILD", "UNH", "JNJ",
    # Financials & Industrial Leaders
    "JPM", "GS", "MS", "V", "MA", "CAT", "GE",
    # China Tech & Global Movers
    "BABA", "PDD", "BIDU",
    # Meme & Momentum Breakout Favorites
    "GME", "AMC", "DJT", "RIVN", "LCID"
]

MEGA_CAP_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "GLD", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", 
    "TSLA", "AVGO", "AMD", "NFLX", "TSM", "ASML", "LLY", "NVO", "LMT", "QCOM", "NOW", "ADBE",
    "COST", "WMT", "JPM", "GS", "V", "MA", "UNH", "CAT", "GE", "DIS", "BA"
}

def get_asset_tier_info(ticker: str) -> dict:
    """
    Returns the classification tier and minimum expected move threshold for a ticker.
    - Mega-Caps / Core Indices: Minimum 4.0% - 5.0% expected move
    - High-Beta Mid-Caps & Runners: Minimum 18.0% - 30.0% expected move
    """
    sym = ticker.upper()
    if sym in MEGA_CAP_TICKERS:
        return {
            "tier": "MEGA-CAP",
            "min_move_pct": 4.5,
            "label": "🏢 Mega-Cap Leader",
            "target_desc": "5%+ Expansion Target"
        }
    else:
        return {
            "tier": "MID-CAP",
            "min_move_pct": 18.0,
            "label": "🚀 High-Beta Mid-Cap Runner",
            "target_desc": "20-30%+ Explosive Target"
        }

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
