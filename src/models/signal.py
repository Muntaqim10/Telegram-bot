from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TradeSignal:
    ticker: str
    price: float
    signal_direction: str
    strategy_type: str
    timestamp: Optional[str] = None
    
    # AI / Sentinel Data
    z_vol: float = 0.0
    is_whale: bool = False
    rev_growth: float = 0.0
    win_probability: float = 0.0
    xgb_win_prob: float = 0.0
    sentinel_verdict: str = "UNKNOWN"
    context_score: str = ""
    catalyst: str = ""
    
    # Technicals
    historical_win_rate: float = 0.0
    rsi: float = 0.0
    
    # Exits
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
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
