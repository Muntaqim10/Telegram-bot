import aiohttp
import logging
import asyncio
import time
from typing import Dict, Any, Optional
from src.data.iv_history import IVHistoryTracker

log = logging.getLogger(__name__)

class OptionsPricer:
    """
    Evaluates real-time options chains to ensure the contract is not overpriced,
    meets liquidity thresholds, is affordable for small accounts, and mathematically
    viable based on projected take-profit (breakeven gate).
    """
    def __init__(self, tradier_token: str):
        self.token = tradier_token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }
        self._chain_cache = {}
        self.iv_tracker = IVHistoryTracker()
        
    async def find_optimal_contract(
        self, ticker: str, expiration: str, target_strike: float, 
        option_type: str, take_profit: float, session=None,
        days_since_earnings: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Scans the chain for an optimal contract near the target strike that passes:
        1. Liquidity (OI/Volume + tight spread)
        2. Affordability (ask <= $2.50)
        3. Breakeven (target take_profit clears the strike + premium)
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
            "reason": "",
            "target_strike": target_strike
        }
        
        try:
            # 1. Check Short-Lived Cache (15-second TTL)
            cache_key = (ticker, expiration)
            cached_entry = self._chain_cache.get(cache_key)
            if cached_entry:
                cached_time, cached_data = cached_entry
                if time.time() - cached_time < 15.0:
                    data = cached_data
                else:
                    data = None
            else:
                data = None

            # 2. Fetch with Exponential Backoff
            if not data:
                max_retries = 3
                
                async def perform_request(s):
                    async with s.get(url, headers=self.headers, timeout=10.0) as resp:
                        if resp.status == 429:
                            return 429, None
                        if resp.status != 200:
                            if 400 <= resp.status < 500:
                                return resp.status, None
                            raise aiohttp.ClientError(f"HTTP {resp.status}")
                        return 200, await resp.json()

                for attempt in range(1, max_retries + 1):
                    try:
                        if session:
                            status, resp_data = await perform_request(session)
                        else:
                            async with aiohttp.ClientSession() as s:
                                status, resp_data = await perform_request(s)
                                
                        if status == 429:
                            if attempt < max_retries:
                                sleep_time = 2 ** (attempt - 1)
                                log.warning(f"Tradier rate limit (429) hit for {ticker}. Retrying in {sleep_time}s (Attempt {attempt}/{max_retries})")
                                await asyncio.sleep(sleep_time)
                                continue
                            else:
                                log.error(f"Tradier rate limit exhausted for {ticker} after {max_retries} attempts.")
                                return result
                                
                        if status != 200:
                            log.error(f"Tradier API error {status} for {ticker}. Aborting.")
                            return result

                        data = resp_data
                        self._chain_cache[cache_key] = (time.time(), data)
                        break # Success, exit retry loop
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt < max_retries:
                            sleep_time = 2 ** (attempt - 1)
                            log.warning(f"Transient network error fetching chain for {ticker}: {e}. Retrying in {sleep_time}s...")
                            await asyncio.sleep(sleep_time)
                        else:
                            log.error(f"Failed to fetch options chain for {ticker} after {max_retries} attempts: {e}")
                            return result

            if not data:
                return result
                
            options = data.get("options", {}).get("option", [])
            if not isinstance(options, list):
                options = [options]
                
            # Calculate GEX (Gamma Exposure), Call vs Put Money Flow & IV Skew across the chain
            best_gex_strike = target_strike
            max_gex = 0.0
            total_call_gex = 0.0
            total_put_gex = 0.0
            total_call_dollar_flow = 0.0
            total_put_dollar_flow = 0.0
            call_iv_list = []
            put_iv_list = []

            valid_contracts = [] # To store contracts matching our option_type for later filtering

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
                    if o_type == option_type.lower() and target_strike * 0.95 <= o_strike <= target_strike * 1.05:
                        if gex > max_gex:
                            max_gex = gex
                            best_gex_strike = o_strike
                elif o_type == "put":
                    total_put_gex += gex
                    total_put_dollar_flow += dflow
                    if iv > 0: put_iv_list.append(iv)
                    if o_type == option_type.lower() and target_strike * 0.95 <= o_strike <= target_strike * 1.05:
                        if gex > max_gex:
                            max_gex = gex
                            best_gex_strike = o_strike

                if o_type == option_type.lower():
                    valid_contracts.append(opt)

            result["gex_target_strike"] = best_gex_strike
            result["gex_value"] = max_gex
            result["gex_confidence"] = "HIGH" if (days_since_earnings is not None and 0 <= days_since_earnings <= 3) else "STANDARD"
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

            if option_type.lower() == "call":
                result["flow_ratio"] = total_call_dollar_flow / (total_put_dollar_flow + 1.0) 
            else:
                result["flow_ratio"] = total_put_dollar_flow / (total_call_dollar_flow + 1.0)

            if not valid_contracts:
                log.warning(f"No {option_type} contracts found for {ticker} at {expiration}.")
                return result

            # Sort contracts by distance to target_strike (preferring strikes slightly closer to ITM if equidistant)
            # For calls: prefer lower strike (closer to ITM) if distance is equal. For puts: prefer higher.
            def sort_key(c):
                s = float(c.get("strike", 0.0))
                dist = abs(s - target_strike)
                # Penalty for being further OTM to break ties
                otm_penalty = s if option_type.lower() == "call" else -s 
                return (dist, otm_penalty)

            valid_contracts.sort(key=sort_key)

            optimal_contract = None
            fallback_contract = None # Most liquid contract that fails affordability/breakeven
            fail_reason = ""

            for contract in valid_contracts:
                strike = float(contract.get("strike", 0.0))
                bid = contract.get("bid", 0.0) or 0.0
                ask = contract.get("ask", 0.0) or 0.0
                mid = (bid + ask) / 2.0
                vol = contract.get("volume", 0) or 0
                oi = contract.get("open_interest", 0) or 0

                # Gate 1: Liquidity Filter
                spread = ask - bid
                spread_ratio = spread / mid if mid > 0 else 1.0
                is_liquid = (oi >= 50 or vol >= 20) and (spread_ratio <= 0.30)
                
                if not is_liquid:
                    continue # Skip illiquid contracts entirely
                    
                # If it's liquid, save it as a fallback in case nothing passes affordability/breakeven
                if not fallback_contract:
                    fallback_contract = contract

                # Gate 2: Affordability Filter
                import os
                default_max = max(2.50, strike * 0.015) # Dynamic: allows up to $7.50 on a $500 stock, or $2.50 on a $50 stock
                max_premium = float(os.getenv("MAX_PREMIUM_DOLLARS", str(default_max)))
                
                if ask > max_premium:
                    fail_reason = f"Premium too expensive (${ask:.2f} > ${max_premium:.2f} limit)."
                    continue
                    
                # Gate 3: Breakeven Gate
                if option_type.lower() == "call":
                    breakeven = strike + ask
                    if take_profit < breakeven:
                        fail_reason = f"Breakeven (${breakeven:.2f}) exceeds TP (${take_profit:.2f})."
                        continue
                else: # Put
                    breakeven = strike - ask
                    if take_profit > breakeven:
                        fail_reason = f"Breakeven (${breakeven:.2f}) below TP (${take_profit:.2f})."
                        continue
                        
                # If we get here, it passed all gates!
                optimal_contract = contract
                break

            final_contract = optimal_contract

            if not final_contract:
                if fallback_contract:
                    # We found a liquid contract but it failed affordability or breakeven
                    final_contract = fallback_contract
                    result["verdict"] = "UNTRADEABLE AT SIZE"
                    result["reason"] = f"Valid setup, not tradeable at your size. {fail_reason}"
                else:
                    # Literally zero liquid contracts found
                    result["verdict"] = "POOR VALUE"
                    result["reason"] = "No liquid options available for this setup."
                    return result

            # Extract Data for the final contract
            result["target_strike"] = float(final_contract.get("strike", 0.0))
            result["bid"] = final_contract.get("bid", 0.0)
            result["ask"] = final_contract.get("ask", 0.0)
            
            midpoint = (result["bid"] + result["ask"]) / 2.0
            
            greeks = final_contract.get("greeks", {})
            if greeks:
                result["delta"] = greeks.get("delta", 0.0)
                result["theta"] = greeks.get("theta", 0.0)
                result["iv"] = greeks.get("smv_vol", greeks.get("implied_volatility", 0.0))
                
            self.iv_tracker.record_iv(ticker, result["iv"])
            iv_rank = self.iv_tracker.get_iv_rank(ticker, result["iv"])
            
            # Unusual Flow Heuristic (e.g. Volume is 3x Open Interest)
            volume = final_contract.get("volume", 0)
            open_interest = final_contract.get("open_interest", 0)
            dollar_flow = volume * midpoint * 100
            
            MIN_ABSOLUTE_VOLUME_FOR_WHALE = 100
            is_unusual_flow = False
            if (open_interest > 0 
                and volume > (open_interest * 3) 
                and volume >= MIN_ABSOLUTE_VOLUME_FOR_WHALE
                and dollar_flow > 50000):
                is_unusual_flow = True
            
            result["is_whale"] = is_unusual_flow
            result["volume"] = volume
            result["open_interest"] = open_interest
            result["dollar_flow"] = dollar_flow
            
            if midpoint > 0:
                result["premium_ratio"] = result["delta"] / midpoint
            
            # Evaluation Logic for contracts that passed the hard gates
            if result["verdict"] == "UNKNOWN": # i.e. it wasn't flagged as UNTRADEABLE AT SIZE
                if iv_rank is not None:
                    result["iv_rank"] = iv_rank
                    if iv_rank > 80:
                        result["verdict"] = "OVERPRICED - HIGH IV RANK"
                        result["reason"] = f"IV Rank is {iv_rank:.0f}/100 for this ticker -- historically expensive."
                    elif result["premium_ratio"] < 0.10 and midpoint > 1.0:
                        result["verdict"] = "POOR VALUE"
                        result["reason"] = f"Paying ${midpoint:.2f} for only {result['delta']:.2f} Delta."
                    else:
                        result["verdict"] = "FAIR VALUE"
                        result["reason"] = "Pricing, liquidity, and breakeven are optimal."
                else:
                    # Not enough IV history yet for this ticker -- fall back to the 
                    # old absolute threshold rather than skip the check entirely
                    if result["iv"] > 1.20:
                        result["verdict"] = "OVERPRICED - HIGH IV"
                        result["reason"] = f"IV is bloated at {result['iv']*100:.1f}%. High risk of IV Crush."
                    elif result["premium_ratio"] < 0.10 and midpoint > 1.0:
                        result["verdict"] = "POOR VALUE"
                        result["reason"] = f"Paying ${midpoint:.2f} for only {result['delta']:.2f} Delta."
                    else:
                        result["verdict"] = "FAIR VALUE"
                        result["reason"] = "Pricing, liquidity, and breakeven are optimal."
                
            return result
                
        except Exception as e:
            log.error(f"OptionsPricer.find_optimal_contract failed for {ticker} (Target: {target_strike} {option_type}, Exp: {expiration}): {e}")
            return result
