import aiohttp
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from src.data.iv_history import IVHistoryTracker

log = logging.getLogger(__name__)

class OptionsPricer:
    """
    Evaluates real-time options chains to ensure the contract is high conviction,
    targets In-The-Money (ITM ~0.75 Delta) for high intrinsic leverage and low theta,
    meets tight liquidity thresholds, and mathematically viable based on take-profit.
    """
    def __init__(self, tradier_token: str):
        self.token = tradier_token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }
        self._chain_cache = {}
        self.iv_tracker = IVHistoryTracker()

    async def get_target_expiration(self, ticker: str, min_dte: int = 1, max_dte: int = 30, session=None) -> str:
        """
        Dynamically queries Tradier for available option expirations for the ticker across 1-30 DTE.
        Handles frequent 3-day / Mon-Wed-Fri short-cycle expirations (e.g. NVDA, SPY, QQQ, TSLA)
        by selecting the nearest active expiration with DTE >= min_dte.
        """
        url = f"https://api.tradier.com/v1/markets/options/expirations?symbol={ticker}&includeAllRoots=true"
        try:
            expirations_list = []
            if session:
                async with session.get(url, headers=self.headers, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        dates = data.get("expirations", {}).get("date", [])
                        expirations_list = dates if isinstance(dates, list) else [dates] if dates else []
            else:
                async with aiohttp.ClientSession() as s:
                    async with s.get(url, headers=self.headers, timeout=5.0) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            dates = data.get("expirations", {}).get("date", [])
                            expirations_list = dates if isinstance(dates, list) else [dates] if dates else []
            
            today = datetime.now().date()
            valid_exps = []
            for exp_str in expirations_list:
                try:
                    exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = (exp_dt - today).days
                    if min_dte <= dte <= max_dte:
                        valid_exps.append((dte, exp_str))
                except ValueError:
                    continue
            
            if valid_exps:
                valid_exps.sort(key=lambda x: x[0])
                log.info(f"[{ticker}] Dynamic expiration selected: {valid_exps[0][1]} ({valid_exps[0][0]} DTE)")
                return valid_exps[0][1]
            elif expirations_list:
                # If nothing in [min_dte, max_dte], select the nearest available real exchange expiration
                log.info(f"[{ticker}] No expirations in {min_dte}-{max_dte} DTE. Using nearest real exchange date: {expirations_list[0]}")
                return expirations_list[0]
        except Exception as e:
            log.warning(f"Failed to fetch dynamic expirations for {ticker}: {e}.")
            
        # Absolute last resort fallback: next Friday
        now = datetime.now()
        target_date = now + timedelta(days=max(1, min_dte))
        days_to_friday = (4 - target_date.weekday()) % 7
        target_expiration = target_date + timedelta(days=days_to_friday)
        return target_expiration.strftime("%Y-%m-%d")
        
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

            # Sort contracts by proximity to target 0.75 Delta (In-The-Money high conviction)
            def sort_key(c):
                greeks = c.get("greeks", {}) or {}
                delta = abs(greeks.get("delta", 0.0) or 0.0)
                if delta > 0:
                    # Primary preference: closest to 0.75 Delta (ITM)
                    delta_diff = abs(delta - 0.75)
                    return (0, delta_diff)
                else:
                    # Fallback if no Greeks: strike distance to estimated ITM strike (5% ITM)
                    s = float(c.get("strike", 0.0))
                    itm_target = target_strike * 0.95 if option_type.lower() == "call" else target_strike * 1.05
                    return (1, abs(s - itm_target))

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

                # Gate 1: Liquidity Filter (Tight spread: max 18% spread ratio OR max $0.35 spread for liquid high-dollar stocks)
                spread = ask - bid
                spread_ratio = spread / mid if mid > 0 else 1.0
                is_tight_spread = (spread_ratio <= 0.18) or (spread <= 0.35 and mid >= 3.0)
                is_liquid = (oi >= 50 or vol >= 20) and is_tight_spread
                
                if not is_liquid:
                    continue # Skip illiquid / wide spread contracts entirely
                    
                # If it's liquid, save it as a fallback in case nothing passes affordability/breakeven
                if not fallback_contract:
                    fallback_contract = contract

                # Gate 2: Affordability Filter (Sized for ITM intrinsic value)
                import os
                default_max = max(15.0, strike * 0.10) # Allows up to $15 on volatile tickers to accommodate ITM intrinsic value
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
                    result["reason"] = "No liquid options with tight bid-ask spread available for this setup."
                    return result

            # Extract Data for the final contract
            result["target_strike"] = float(final_contract.get("strike", 0.0))
            result["occ_symbol"] = final_contract.get("symbol", "")
            result["expiration"] = final_contract.get("expiration_date", expiration)
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
                result["premium_ratio"] = abs(result["delta"]) / midpoint
                result["theta_ratio"] = abs(result["theta"]) / midpoint
            else:
                result["premium_ratio"] = 0.0
                result["theta_ratio"] = 0.0

            # Secondary Scan: Scan for Optional High-Leverage OTM Runner (~0.35 Delta)
            otm_contracts = []
            for contract in valid_contracts:
                greeks = contract.get("greeks", {}) or {}
                delta = abs(greeks.get("delta", 0.0) or 0.0)
                strike = float(contract.get("strike", 0.0))
                bid = contract.get("bid", 0.0) or 0.0
                ask = contract.get("ask", 0.0) or 0.0
                vol = contract.get("volume", 0) or 0
                oi = contract.get("open_interest", 0) or 0
                spread = ask - bid
                mid = (bid + ask) / 2.0
                spread_ratio = spread / mid if mid > 0 else 1.0

                is_liquid_otm = (oi >= 50 or vol >= 25) and (spread_ratio <= 0.25 or spread <= 0.35)
                if not is_liquid_otm or ask <= 0.10:
                    continue

                if 0.20 <= delta <= 0.45:
                    if option_type.lower() == "call" and (strike + ask) <= take_profit * 1.03:
                        otm_contracts.append((abs(delta - 0.35), contract))
                    elif option_type.lower() == "put" and (strike - ask) >= take_profit * 0.97:
                        otm_contracts.append((abs(delta - 0.35), contract))

            if otm_contracts:
                otm_contracts.sort(key=lambda x: x[0])
                best_otm = otm_contracts[0][1]
                otm_greeks = best_otm.get("greeks", {}) or {}
                otm_ask = best_otm.get("ask", 0.0) or 0.0
                result["otm_runner"] = {
                    "strike": float(best_otm.get("strike", 0.0)),
                    "occ_symbol": best_otm.get("symbol", ""),
                    "bid": best_otm.get("bid", 0.0),
                    "ask": otm_ask,
                    "delta": otm_greeks.get("delta", 0.0),
                    "theta": otm_greeks.get("theta", 0.0),
                    "iv": otm_greeks.get("smv_vol", otm_greeks.get("implied_volatility", 0.0)),
                    "opt_tp": round(otm_ask * 1.80, 2),
                    "opt_sl": round(otm_ask * 0.50, 2)
                }
            
            # Evaluation Logic for contracts that passed the hard gates
            if result["verdict"] == "UNKNOWN": # i.e. it wasn't flagged as UNTRADEABLE AT SIZE
                delta_abs = abs(result.get("delta", 0.0))
                theta_ratio = result.get("theta_ratio", 0.0)
                
                # High Conviction In-The-Money (ITM) Greek Gates (Target ~0.75 Delta, Accept 0.60 - 0.90 Delta)
                if delta_abs > 0 and (delta_abs < 0.60 or delta_abs > 0.90):
                    result["verdict"] = "POOR VALUE"
                    result["reason"] = f"Delta ({result['delta']:.2f}) outside target ITM range (0.60 - 0.90 Delta). Target is ~0.75 Delta."
                elif theta_ratio > 0.15:
                    result["verdict"] = "POOR VALUE"
                    result["reason"] = f"Excessive Theta decay (${abs(result['theta']):.2f}/day, {theta_ratio*100:.1f}% of premium). Maximum allowed is 15%/day."
                elif iv_rank is not None:
                    result["iv_rank"] = iv_rank
                    if iv_rank > 80:
                        result["verdict"] = "OVERPRICED - HIGH IV RANK"
                        result["reason"] = f"IV Rank is {iv_rank:.0f}/100 for this ticker -- historically expensive."
                    else:
                        result["verdict"] = "FAIR VALUE"
                        result["reason"] = "Pricing, tight spread liquidity, 0.75 Delta ITM Greeks, and breakeven are optimal."
                else:
                    # Not enough IV history yet for this ticker -- fall back to absolute threshold
                    if result["iv"] > 1.20:
                        result["verdict"] = "OVERPRICED - HIGH IV"
                        result["reason"] = f"IV is bloated at {result['iv']*100:.1f}%. High risk of IV Crush."
                    else:
                        result["verdict"] = "FAIR VALUE"
                        result["reason"] = "Pricing, tight spread liquidity, 0.75 Delta ITM Greeks, and breakeven are optimal."
                
            return result
                
        except Exception as e:
            log.error(f"OptionsPricer.find_optimal_contract failed for {ticker} (Target: {target_strike} {option_type}, Exp: {expiration}): {e}")
            return result
