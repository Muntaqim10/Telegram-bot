from dataclasses import dataclass
from typing import Optional

@dataclass
class TradeSignal:
    ticker: str
    price: float
    signal_direction: str
    strategy_type: str
    timestamp: Optional[str] = None
    
    # AI / Sentinel Data
    is_whale: bool = False
    win_probability: float = 0.0
    xgb_win_prob: float = 0.0
    sentinel_verdict: str = "UNKNOWN"
    conviction: str = "🔴 LOW"  # Tiered: 🟢 HIGH / 🟡 MEDIUM / 🔴 LOW
    context_score: str = ""
    catalyst: str = ""
    
    # Real Technicals (computed from live data)
    vwap_ratio: float = 1.0
    volume: float = 0.0
    warning_tag: Optional[str] = None  # Soft-fail warnings (e.g. VWAP overextension)
    strategy_suggestion: Optional[str] = None  # Alt strategy hint (e.g. debit spread when IV is rich)
    
    # Exits
    stop_loss: float = 0.0
    take_profit: float = 0.0
    asset_tier: str = ""
    expected_move_pct: float = 0.0
    
    # Options / Pricing
    option_ask: float = 0.0
    option_bid: float = 0.0
    option_tp_pct: float = 0.0
    option_sl_pct: float = 0.0
    delta: float = 0.0
    theta: float = 0.0
    iv: float = 0.0
    flow_bias: str = "BALANCED FLOW"
    call_dollar_flow: float = 0.0
    put_dollar_flow: float = 0.0
    net_gex: float = 0.0
    
    # Expiration details
    expiry: Optional[str] = None
    target_strike: Optional[float] = None
    option_type: Optional[str] = None
    intraday_expiry: Optional[str] = None
    intraday_strike: Optional[float] = None
    intraday_option_type: Optional[str] = None
    timeframe_target: str = "Intraday"
    pricing_verdict: str = ""
    pricing_reason: str = ""
    
    # Earnings risk
    earnings_risk: bool = False
    earnings_date: Optional[str] = None
    gex_confidence: str = "STANDARD"
