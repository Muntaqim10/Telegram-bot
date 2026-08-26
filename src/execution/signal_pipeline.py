import logging
from datetime import datetime
from typing import Dict, Any
from src.models.signal import TradeSignal
from src.data.earnings_calendar import EarningsCalendar
from src.backtest.stats_reader import BacktestStatsReader

log = logging.getLogger("rallyhunter.pipeline")

class SignalPipeline:
    def __init__(self, news_fetcher, sentiment_analyzer, xgb_model, risk_manager, options_pricer, alerts, donchian_strategy=None):
        self.news_fetcher = news_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        self.xgb_model = xgb_model
        self.risk_manager = risk_manager
        self.options_pricer = options_pricer
        self.alerts = alerts
        self.donchian_strategy = donchian_strategy
        self.earnings_calendar = EarningsCalendar()
        self.backtest_reader = BacktestStatsReader()
        
        from src.data.flashalpha_client import FlashAlphaClient
        self.flashalpha = FlashAlphaClient()
        from src.strategy.timeframe_confluence import TimeframeConfluenceEngine
        self.tf_engine = TimeframeConfluenceEngine()

    async def process_signal(self, signal: Dict[str, Any], spy_vwap_ratio: float = 1.0, session=None):
        """Passes the raw signal through AI filters before alerting."""
        ticker = signal["ticker"]
        direction = signal["direction"]

        # 1. Market Bias Tracking
        if direction == "Long":
            log.info(f"🟢 CALL BIAS ACTIVE: {ticker} Long breakout setup evaluated without SPY block.")
        else:
            log.info(f"🔴 {ticker} Short breakdown setup evaluated with risk management.")

        # 1.5 VWAP Overextension Check (soft-fail: tag, don't suppress)
        vwap_ratio = signal.get("vwap_ratio", 1.0)
        warnings = []
        
        if direction == "Long" and vwap_ratio > 1.025:
            warnings.append(f"Extended (+{(vwap_ratio - 1)*100:.1f}% > VWAP)")
            log.info(f"⚠️ {ticker} Call: Price overextended ({vwap_ratio:.4f}). Tagging alert, not blocking.")
        elif direction == "Short" and vwap_ratio < 0.975:
            warnings.append(f"Extended ({(vwap_ratio - 1)*100:.1f}% < VWAP)")
            log.info(f"⚠️ {ticker} Put: Price overextended ({vwap_ratio:.4f}). Tagging alert, not blocking.")

        # 2. AI Sentiment Check (Blind)
        headlines = await self.news_fetcher.get_headlines(ticker, session=session)
        sent_score, catalyst, is_whale, sent_conf = await self.sentiment_analyzer.score_headlines(ticker, headlines)

        if direction == "Short" and sent_score > 0.65:
            warnings.append(f"Fighting positive news ({sent_score:.2f} score)")
            log.info(f"⚠️ {ticker} Short: Strong positive news ({sent_score:.2f}). Tagging alert, not blocking.")
        elif direction == "Long" and sent_score < 0.35:
            warnings.append(f"Fighting negative news ({sent_score:.2f} score)")
            log.info(f"⚠️ {ticker} Long: Strong negative news ({sent_score:.2f}). Tagging alert, not blocking.")
            
        final_warning_tag = "⚠️ " + " | ".join(warnings) if warnings else None

        # 3. XGBoost Sentinel Check → Conviction Tier (never a silent kill)
        sig_ts = signal.get("timestamp")
        if sig_ts is None or not hasattr(sig_ts, "hour"):
            sig_ts = datetime.now()

        features = {
            "relative_volume": float(signal.get("relative_volume") or signal.get("z_vol") or 1.5),
            "rsi_14": float(signal.get("rsi_14", 55.0)),
            "chop_14": float(signal.get("chop_14", 45.0)),
            "expected_move_pct": float(signal.get("expected_move_pct", 4.0)),
            "hist_vol_20": float(signal.get("hist_vol_20", 0.35)),
            "sma20_ratio": float(signal.get("sma20_ratio", vwap_ratio)),
            "sma_spread": float(signal.get("sma_spread", 0.02)),
            "breakout_pct": float(signal.get("breakout_pct", 0.01)),
            "direction_code": 1 if direction == "Long" else 0
        }
        xgb_result = self.xgb_model.validate_setup(features)
        win_prob = xgb_result['win_prob']
        verdict = xgb_result['verdict']

        # Conviction tiers based on model output & sentiment
        if win_prob >= 0.70:
            conviction = "🟢 HIGH"
        elif win_prob >= 0.45 or sent_score >= 0.55:
            conviction = "🟡 MEDIUM"
        else:
            conviction = "🔴 LOW"

        log.info(f"[{ticker}] AI Filters: Sentiment={sent_score:.2f}, XGB={verdict} ({win_prob:.2f}), Conviction={conviction}")
        
        # Enforce Fakeout Filter: Suppress only low-probability traps or setups fighting sharp opposite news
        if win_prob < 0.40 or (direction == "Long" and sent_score < 0.30) or (direction == "Short" and sent_score > 0.70):
            log.warning(f"[{ticker}] Alert suppressed by Fakeout Filter: XGB={verdict} ({win_prob:.2f}), Sentiment={sent_score:.2f}. Conviction: {conviction}")
            return
            
        log.info(f"✅ {ticker} {direction} Conviction: {conviction}. Dispatching Alert.")

        # Extract breakout level for fast thesis invalidation (Momentum Movers, ORB, Donchian, CRB)
        invalidation_level = signal.get("invalidation_level") or signal.get("breakout_level")
        if invalidation_level is None:
            if direction == "Long":
                invalidation_level = signal.get("orb_high") or signal.get("donchian_high") or signal.get("crb_high")
            else:
                invalidation_level = signal.get("orb_low") or signal.get("donchian_low") or signal.get("crb_low")
        if invalidation_level is not None:
            try:
                invalidation_level = float(invalidation_level)
            except (ValueError, TypeError):
                invalidation_level = None

        # Extract natively computed SL/TP if they exist (e.g., from Donchian signals)
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        cat_type = signal.get("catalyst_type", "Standard Breakout")

        if stop_loss is not None and take_profit is not None:
            risk_levels = self.risk_manager.add_position(
                ticker, signal["entry_price"], initial_atr=0.0, direction=signal["direction"],
                stop_loss=stop_loss, take_profit=take_profit, invalidation_level=invalidation_level,
                catalyst_type=cat_type
            )
        else:
            # ORB Signals: attempt to fetch cached real ATR, otherwise fallback with warning
            cached_atr = None
            if self.donchian_strategy and hasattr(self.donchian_strategy, "atr_cache"):
                cached_atr = self.donchian_strategy.atr_cache.get(ticker)
            
            import pandas as pd
            if cached_atr and not pd.isna(cached_atr):
                risk_levels = self.risk_manager.add_position(
                    ticker, signal["entry_price"], initial_atr=cached_atr, 
                    direction=signal["direction"], invalidation_level=invalidation_level,
                    catalyst_type=cat_type
                )
            else:
                import logging
                logging.getLogger("rallyhunter.pipeline").warning(f"[{ticker}] No cached daily ATR available. Falling back to default ATR=1.5")
                risk_levels = self.risk_manager.add_position(
                    ticker, signal["entry_price"], initial_atr=1.5, 
                    direction=signal["direction"], invalidation_level=invalidation_level,
                    catalyst_type=cat_type
                )

        # 3.5 Tiered Velocity & Expected Move Magnitude Gate
        from src.data.dynamic_scanner import get_asset_tier_info
        tier_info = get_asset_tier_info(ticker)
        entry_p = float(signal["entry_price"])
        tp_p = float(risk_levels["take_profit"])
        expected_move_pct = (abs(tp_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0
        
        min_required_move = tier_info["min_move_pct"]
        z_vol_val = signal.get("z_vol", 0.0)
        if isinstance(z_vol_val, str):
            try: z_vol_val = float(z_vol_val)
            except: z_vol_val = 0.0

        if expected_move_pct < min_required_move and z_vol_val < 1.8:
            log.warning(
                f"[{ticker}] Alert suppressed by Velocity Gate: {tier_info['tier']} expected move ({expected_move_pct:.1f}%) "
                f"is below the minimum required threshold ({min_required_move}%)."
            )
            self.risk_manager.remove_position(ticker)
            return

        # 4. Options Pricing Check (Targeting 14-21 DTE for optimal swing leverage and low theta decay)
        expiration = await self.options_pricer.get_target_expiration(ticker, min_dte=14, max_dte=21, session=session)
        target_strike = float(round(signal["entry_price"])) # Spot reference (OptionsPricer optimizes to ~0.75 Delta ITM)

        # 4.1 Evaluate Multi-Timeframe Triad Confluence (Weekly/Daily/4H for Weeklies vs Daily/4H/1H for 3-Day)
        try:
            exp_dt = datetime.strptime(expiration, "%Y-%m-%d").date()
            target_dte = (exp_dt - datetime.now().date()).days
        except Exception:
            target_dte = 7

        intraday_ctx = {
            "vwap": signal.get("vwap", entry_p),
            "ema9": signal.get("ema9", entry_p),
            "hod": signal.get("hod", entry_p * 1.02),
            "lod": signal.get("lod", entry_p * 0.98),
        }

        tf_matrix = await self.tf_engine.evaluate_confluence(
            ticker=ticker,
            spot_price=entry_p,
            direction=direction,
            dte=target_dte,
            intraday_state=intraday_ctx,
            session=session
        )

        if not tf_matrix.get("concordant", True):
            log.warning(
                f"[{ticker}] Alert suppressed by Timeframe Confluence ({tf_matrix['cycle_name']}): "
                f"Triad discordance across {tf_matrix['confluence_summary']}."
            )
            self.risk_manager.remove_position(ticker)
            return

        if tf_matrix.get("data_quality_warning"):
            quality_note = "⚠️ Timeframe confluence evaluated on incomplete historical data"
            if final_warning_tag:
                final_warning_tag += f" | {quality_note}"
            else:
                final_warning_tag = quality_note
            log.warning(f"[{ticker}] Timeframe confluence data quality warning flagged. Appending note to alert.")

        opt_type = "put" if direction == "Short" else "call"

        next_earnings = await self.earnings_calendar.get_next_earnings_date(ticker, session=session)
        earnings_in_window = False
        if next_earnings:
            try:
                earnings_dt = datetime.strptime(next_earnings, "%Y-%m-%d").date()
                expiration_dt = datetime.strptime(expiration, "%Y-%m-%d").date()
                today_dt = datetime.now().date()
                earnings_in_window = today_dt <= earnings_dt <= expiration_dt
            except ValueError:
                earnings_in_window = False
        
        last_earnings = await self.earnings_calendar.get_last_earnings_date(ticker, session=session)
        days_since = self.earnings_calendar.days_since_earnings(last_earnings)

        pricing_data = await self.options_pricer.find_optimal_contract(
            ticker=ticker,
            expiration=expiration,
            target_strike=target_strike,
            option_type=opt_type,
            take_profit=risk_levels["take_profit"],
            session=session,
            days_since_earnings=days_since
        )
        
        # Attach real option pricing (ask, delta, theta) to the registered position in RiskManager
        self.risk_manager.attach_option_pricing(
            ticker=ticker,
            option_entry_price=pricing_data.get("ask", 0.0),
            option_entry_delta=pricing_data.get("delta", 0.0),
            option_entry_theta=pricing_data.get("theta", 0.0)
        )
        
        # Pull the actual chosen strike, which might differ from target_strike due to affordability/liquidity gates
        actual_strike = pricing_data.get("target_strike", target_strike)

        if direction == "Long" and pricing_data.get("flow_bias") == "BEARISH PUT FLOW" and pricing_data.get("put_dollar_flow", 0) > pricing_data.get("call_dollar_flow", 0) * 2.5:
            log.warning(f"[{ticker}] Long alert suppressed: Heavy Bearish Put Flow.")
            self.risk_manager.remove_position(ticker)
            return
        elif direction == "Short" and pricing_data.get("flow_bias") == "BULLISH CALL FLOW" and pricing_data.get("call_dollar_flow", 0) > pricing_data.get("put_dollar_flow", 0) * 2.5:
            log.warning(f"[{ticker}] Short alert suppressed: Heavy Bullish Call Flow.")
            self.risk_manager.remove_position(ticker)
            return

        # 4.5 FlashAlpha Institutional GEX Filter
        fa_data = await self.flashalpha.get_gex_profile(ticker, expiration)
        if fa_data.get("status") == "ok":
            call_wall = fa_data.get("call_wall", 0.0)
            put_wall = fa_data.get("put_wall", 0.0)
            
            # Smart Money Filter Rules
            if direction == "Long" and call_wall > 0 and signal["entry_price"] >= call_wall * 0.99 and signal["entry_price"] <= call_wall * 1.01:
                log.warning(f"[{ticker}] Long alert suppressed by FlashAlpha: Price ({signal['entry_price']}) is hitting the Market Maker Call Wall ({call_wall}).")
                self.risk_manager.remove_position(ticker)
                return
            elif direction == "Short" and put_wall > 0 and signal["entry_price"] <= put_wall * 1.01 and signal["entry_price"] >= put_wall * 0.99:
                log.warning(f"[{ticker}] Short alert suppressed by FlashAlpha: Price ({signal['entry_price']}) is bouncing off the Market Maker Put Wall ({put_wall}).")
                self.risk_manager.remove_position(ticker)
                return
                
            # Embed wall context into the alert
            if call_wall > 0 and put_wall > 0:
                wall_text = f"🛡️ MM Walls: Call ${call_wall} | Put ${put_wall}"
                final_warning_tag = f"{final_warning_tag} | {wall_text}" if final_warning_tag else wall_text
        elif fa_data.get("status") == "rate_limited":
            final_warning_tag = f"{final_warning_tag} | ⚠️ GEX check skipped (rate limit)" if final_warning_tag else "⚠️ GEX check skipped (rate limit)"
                
        # Verdict handling
        strategy_suggestion = None
        iv_value = pricing_data.get("iv", 0.0)
        pricing_verdict_str = pricing_data.get("verdict", "")
        
        if pricing_verdict_str == "POOR VALUE":
            log.warning(f"[{ticker}] Alert suppressed: {pricing_data['reason']}")
            self.risk_manager.remove_position(ticker)
            return
        elif pricing_verdict_str == "UNTRADEABLE AT SIZE":
            strategy_suggestion = f"⚠️ Setup valid, not tradeable at your size. {pricing_data['reason']}"
            log.info(f"[{ticker}] Untradeable at size. Tagging alert, not blocking.")
        elif pricing_verdict_str in ("OVERPRICED - HIGH IV", "OVERPRICED - HIGH IV RANK") or iv_value > 1.20:
            iv_pct = iv_value * 100
            strategy_suggestion = (
                f"💡 IV is rich ({iv_pct:.0f}%) — premium pricing in continued movement. "
                f"Consider a debit spread (buy ${actual_strike} / sell ${actual_strike + 5 if opt_type == 'call' else actual_strike - 5}) "
                f"to cap vega risk instead of a naked long."
            )
            log.info(f"[{ticker}] High IV ({iv_pct:.0f}%). Tagging with spread suggestion, not blocking.")

        # 4.6 DeepSeek Multi-Domain Synthesis (News + Math + Empirical Backtest Stats)
        bt_stats = self.backtest_reader.get_ticker_stats(ticker)
        math_context = {
            "expected_move_pct": expected_move_pct,
            "target_strike": actual_strike,
            "option_type": opt_type.upper(),
            "option_ask": pricing_data.get("ask", 0.0),
            "delta": pricing_data.get("delta", 0.75),
            "theta": pricing_data.get("theta", 0.0),
            "iv_rank": pricing_data.get("iv_rank"),
            "gex_confidence": pricing_data.get("gex_confidence", "STANDARD"),
            "timeframe_confluence": tf_matrix.get("confluence_summary", "N/A"),
            # Note: is_early_surge is computed downstream in section 5
        }
        
        synth_result = await self.sentiment_analyzer.synthesize_catalyst_and_edge(
            ticker=ticker,
            direction=direction,
            headlines=headlines,
            backtest_stats=bt_stats,
            math_context=math_context
        )
        
        catalyst = synth_result.get("catalyst", catalyst)
        ai_thesis = synth_result.get("ai_thesis", "")
        if synth_result.get("is_whale"):
            is_whale = True
        bt_age = bt_stats.get("backtest_age_days")
        staleness_note = f" ⚠️ (stale: {bt_age:.0f}d old)" if bt_age is not None and bt_age > 7 else ""

        hist_edge_str = ""
        if bt_stats.get("total_trades", 0) > 0:
            hist_edge_str = (
                f"{bt_stats['win_rate']}% Win Rate ({bt_stats['total_trades']} trades | "
                f"+{bt_stats['avg_return_pct']}% Avg ROI | Max +{bt_stats['max_gain_pct']}%)"
                f"{staleness_note}"
            )
        else:
            hist_edge_str = "Initial Live Tracking (Broad Watchlist Dynamic Scan)"

        # Check if setup qualifies for High-Leverage OTM Runner (Rare occasions: DeepSeek + Math + Backtest)
        raw_otm = pricing_data.get("otm_runner")
        otm_qualified = False
        if raw_otm and synth_result.get("otm_qualified", False):
            min_exp = 8.0 if tier_info.get("tier") == "MEGA-CAP" else 20.0
            if xgb_result['win_prob'] >= 0.75 and expected_move_pct >= min_exp:
                otm_qualified = True
                log.info(f"🚀 [{ticker}] Setup QUALIFIED for RARE High-Leverage OTM Runner (${raw_otm.get('strike')} {opt_type.upper()}).")

        # Check for Early Morning Surge High Alert (>=3% Mega-Cap or >=10% Mid-Cap before 10:30 AM EST)
        is_early_surge = signal.get("is_early_surge", False)
        early_surge_desc = signal.get("surge_type", "")
        sig_time = signal.get("timestamp")
        if not is_early_surge:
            try:
                hour = sig_time.hour if hasattr(sig_time, "hour") else 9
                minute = sig_time.minute if hasattr(sig_time, "minute") else 30
            except Exception:
                hour, minute = 9, 30

            is_morning = (hour < 10) or (hour == 10 and minute <= 30)
            is_mega = tier_info.get("tier") == "MEGA-CAP"
            if is_morning:
                if is_mega and expected_move_pct >= 3.0:
                    is_early_surge = True
                    early_surge_desc = f"MEGA-CAP SURGE (+{expected_move_pct:.1f}%)"
                    log.info(f"🚨 [{ticker}] HIGH ALERT: Early Morning {early_surge_desc}")
                elif not is_mega and expected_move_pct >= 10.0:
                    is_early_surge = True
                    early_surge_desc = f"MID-CAP SURGE (+{expected_move_pct:.1f}%)"
                    log.info(f"🚨 [{ticker}] HIGH ALERT: Early Morning {early_surge_desc}")

        # 5. Build TradeSignal Object
        trade_signal = TradeSignal(
            ticker=ticker,
            price=signal["entry_price"],
            signal_direction=signal["direction"],
            strategy_type=signal["catalyst_type"],
            timestamp=signal["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(signal["timestamp"], "strftime") else str(signal["timestamp"]),
            is_whale=is_whale or pricing_data.get("is_whale", False),
            win_probability=xgb_result['win_prob'],
            xgb_win_prob=xgb_result['win_prob'],
            sentinel_verdict=verdict,
            conviction=conviction,
            context_score=tf_matrix.get("confluence_summary", signal.get("tf_confluence", "Intraday ORB")),
            cycle_type=tf_matrix.get("cycle_name", "WEEKLY SWING CYCLE"),
            timeframe_matrix=tf_matrix,
            catalyst=catalyst,
            ai_thesis=ai_thesis,
            historical_edge=hist_edge_str,
            vwap_ratio=signal.get("vwap_ratio", 1.0),
            volume=signal.get("volume", 0.0),
            warning_tag=final_warning_tag,
            strategy_suggestion=strategy_suggestion,
            stop_loss=risk_levels["stop_loss"],
            take_profit=risk_levels["take_profit"],
            asset_tier=tier_info.get("label", ""),
            expected_move_pct=expected_move_pct,
            option_ask=pricing_data.get("ask", 0.0),
            option_bid=pricing_data.get("bid", 0.0),
            option_tp_pct=0.50,
            option_sl_pct=-0.25,
            delta=pricing_data.get("delta", 0.0),
            theta=pricing_data.get("theta", 0.0),
            iv=pricing_data.get("iv", 0.0),
            flow_bias=pricing_data.get("flow_bias", "BALANCED FLOW"),
            call_dollar_flow=pricing_data.get("call_dollar_flow", 0.0),
            put_dollar_flow=pricing_data.get("put_dollar_flow", 0.0),
            net_gex=pricing_data.get("net_gex", 0.0),
            occ_symbol=pricing_data.get("occ_symbol", ""),
            expiry=pricing_data.get("expiration", expiration),
            intraday_expiry=pricing_data.get("expiration", expiration),
            target_strike=actual_strike,
            intraday_strike=actual_strike,
            option_type=opt_type.upper(),
            intraday_option_type=opt_type.upper(),
            pricing_verdict=pricing_data.get("verdict", ""),
            pricing_reason=pricing_data.get("reason", ""),
            earnings_risk=earnings_in_window,
            earnings_date=next_earnings,
            gex_confidence=pricing_data.get("gex_confidence", "STANDARD"),
            otm_runner=raw_otm if otm_qualified else None,
            otm_qualified=otm_qualified,
            is_early_surge=is_early_surge,
            early_surge_desc=early_surge_desc,
            invalidation_level=invalidation_level
        )

        await self.alerts.dispatch_high_conviction(trade_signal)
