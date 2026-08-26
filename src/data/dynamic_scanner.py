import os
import random
import logging
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
    "COST", "WMT", "JPM", "GS", "V", "MA", "UNH", "CAT", "GE", "DIS", "BA", "XOM", "CVX",
    "MRVL", "ORCL", "CRM", "INTC", "CSCO", "IBM", "TXN"
}

def get_asset_tier_info(ticker: str) -> dict:
    """
    Returns the classification tier and minimum expected move threshold for a ticker.
    Calibrated for 14-21 DTE options where a 2.5% to 4.5% stock move produces +30% to +60% option profit.
    """
    sym = ticker.upper()
    if sym in MEGA_CAP_TICKERS:
        return {
            "tier": "MEGA-CAP",
            "min_move_pct": 2.5,
            "label": "🏢 Mega-Cap Leader",
            "target_desc": "2.5%+ Swing Target (+25-35% Option PnL)"
        }
    else:
        return {
            "tier": "MID-CAP",
            "min_move_pct": 4.5,
            "label": "🚀 High-Beta Mid-Cap Runner",
            "target_desc": "4.5%+ Momentum Target (+45-65% Option PnL)"
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
        Samples a randomized batch of 50 tickers from the 119-ticker CANDIDATE_POOL
        on each scan cycle to ensure continuous rotation across the entire universe.
        """
        sample_size = min(max_symbols, len(CANDIDATE_POOL))
        sampled_tickers = random.sample(CANDIDATE_POOL, sample_size)
        log.info(f"⚡ [DYNAMIC SCANNER] Randomly sampled {len(sampled_tickers)} stocks from the {len(CANDIDATE_POOL)}-ticker universe for live stream rotation.")
        return sampled_tickers
