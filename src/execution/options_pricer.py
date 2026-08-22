import aiohttp
import logging
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

class OptionsPricer:
    """
    Evaluates real-time options chains to ensure the contract is not overpriced.
    Calculates Delta-to-Premium ratios and checks for IV Bloat to prevent IV crush.
    """
    def __init__(self, tradier_token: str):
        self.token = tradier_token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }
        
    async def evaluate_contract(self, ticker: str, expiration: str, strike: float, option_type: str) -> Dict[str, Any]:
        """
        Fetches the specific contract's Greeks and premium to evaluate its fairness.
        """
        url = f"https://api.tradier.com/v1/markets/options/chains?symbol={ticker}&expiration={expiration}&greeks=true"
        
        result = {
            "verdict": "UNKNOWN",
            "delta": 0.0,
            "theta": 0.0,
            "iv": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "premium_ratio": 0.0,
            "reason": ""
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status != 200:
                        log.error(f"Tradier options chain failed: {resp.status}")
                        return result
                        
                    data = await resp.json()
                    options = data.get("options", {}).get("option", [])
                    if not isinstance(options, list):
                        options = [options]
                        
                    # Calculate GEX (Gamma Exposure), Call vs Put Money Flow & IV Skew across the chain
                    best_gex_strike = strike
                    max_gex = 0.0
                    total_call_gex = 0.0
                    total_put_gex = 0.0
                    total_call_dollar_flow = 0.0
                    total_put_dollar_flow = 0.0
                    call_iv_list = []
                    put_iv_list = []

                    for opt in options:
                        o_type = opt.get("option_type", "").lower()
                        o_strike = float(opt.get("strike", 0.0))
                        oi = opt.get("open_interest", 0) or 0
                        vol = opt.get("volume", 0) or 0
                        bid = opt.get("bid", 0.0) or 0.0
                        ask = opt.get("ask", 0.0) or 0.0
                        mid = (bid + ask) / 2.0
                        dflow = vol * mid * 100.0
                        
                        greeks = opt.get("greeks", {}) or {}
                        gamma = abs(greeks.get("gamma", 0.0) or 0.0)
                        iv = greeks.get("smv_vol", greeks.get("implied_volatility", 0.0)) or 0.0
                        gex = gamma * oi * 100.0

                        if o_type == "call":
                            total_call_gex += gex
                            total_call_dollar_flow += dflow
                            if iv > 0: call_iv_list.append(iv)
                            # For Call alerts, search for OTM strikes above target within 3% range
                            if o_type == option_type.lower() and strike * 0.99 <= o_strike <= strike * 1.03:
                                if gex > max_gex:
                                    max_gex = gex
                                    best_gex_strike = o_strike
                        elif o_type == "put":
                            total_put_gex += gex
                            total_put_dollar_flow += dflow
                            if iv > 0: put_iv_list.append(iv)
                            # For Put alerts, search for OTM strikes below target within 3% range
                            if o_type == option_type.lower() and strike * 0.97 <= o_strike <= strike * 1.01:
                                if gex > max_gex:
                                    max_gex = gex
                                    best_gex_strike = o_strike

                    result["gex_target_strike"] = best_gex_strike
                    result["gex_value"] = max_gex
                    result["net_gex"] = total_call_gex - total_put_gex
                    result["call_dollar_flow"] = total_call_dollar_flow
                    result["put_dollar_flow"] = total_put_dollar_flow
                    
                    # Money Flow Bias & Skew
                    avg_call_iv = (sum(call_iv_list) / len(call_iv_list)) if call_iv_list else 0.0
                    avg_put_iv = (sum(put_iv_list) / len(put_iv_list)) if put_iv_list else 0.0
                    result["avg_call_iv"] = avg_call_iv
                    result["avg_put_iv"] = avg_put_iv
                    
                    if total_call_dollar_flow > total_put_dollar_flow * 1.2:
                        result["flow_bias"] = "BULLISH CALL FLOW"
                    elif total_put_dollar_flow > total_call_dollar_flow * 1.2:
                        result["flow_bias"] = "BEARISH PUT FLOW"
                    else:
                        result["flow_bias"] = "BALANCED FLOW"

                    result["flow_ratio"] = total_call_dollar_flow / (total_put_dollar_flow + 1.0) if option_type.lower() == "call" else total_put_dollar_flow / (total_call_dollar_flow + 1.0)

                    # Find our specific contract
                    contract = None
                    for opt in options:
                        if abs(opt.get("strike", 0.0) - strike) < 0.01 and opt.get("option_type", "").lower() == option_type.lower():
                            contract = opt
                            break
                            
                    if not contract and best_gex_strike != strike:
                        # Fallback to GEX target strike contract if exact strike was unavailable
                        for opt in options:
                            if abs(opt.get("strike", 0.0) - best_gex_strike) < 0.01 and opt.get("option_type", "").lower() == option_type.lower():
                                contract = opt
                                break

                    if not contract:
                        log.warning(f"Contract {ticker} {strike} {option_type} not found in chain.")
                        return result
                        
                    # Extract Data & Institutional Options Flow (Whale Detection)
                    result["bid"] = contract.get("bid", 0.0)
                    result["ask"] = contract.get("ask", 0.0)
                    volume = contract.get("volume", 0) or 0
                    open_interest = contract.get("open_interest", 0) or 0
                    greeks = contract.get("greeks", {})
                    
                    if greeks:
                        result["delta"] = abs(greeks.get("delta", 0.0))
                        result["theta"] = greeks.get("theta", 0.0)
                        result["iv"] = greeks.get("smv_vol", greeks.get("implied_volatility", 0.0))
                        
                    midpoint = (result["bid"] + result["ask"]) / 2.0
                    dollar_flow = volume * midpoint * 100.0
                    
                    is_unusual_flow = (open_interest > 0 and volume > open_interest * 1.2 and volume >= 250) or (dollar_flow >= 100000.0)
                    result["is_whale"] = is_unusual_flow
                    result["volume"] = volume
                    result["open_interest"] = open_interest
                    result["dollar_flow"] = dollar_flow
                    
                    if midpoint > 0:
                        result["premium_ratio"] = result["delta"] / midpoint
                    
                    # Evaluation Logic
                    # A good contract usually provides at least 0.15 to 0.20 delta per $1 of premium for short-term trades
                    # If IV is > 1.0 (100%), it's highly susceptible to IV crush unless it's a very volatile stock (like TSLA/NVDA, which can handle 1.5).
                    
                    if result["iv"] > 1.20:
                        result["verdict"] = "OVERPRICED - HIGH IV"
                        result["reason"] = f"IV is bloated at {result['iv']*100:.1f}%. High risk of IV Crush."
                    elif result["premium_ratio"] < 0.10 and midpoint > 1.0:
                        result["verdict"] = "POOR VALUE"
                        result["reason"] = f"Paying ${midpoint:.2f} for only {result['delta']:.2f} Delta."
                    else:
                        result["verdict"] = "FAIR VALUE"
                        result["reason"] = "Pricing and Greeks are within normal bounds."
                        
                    return result
                        
        except Exception as e:
            log.error(f"Error evaluating options contract: {e}")
            return result
