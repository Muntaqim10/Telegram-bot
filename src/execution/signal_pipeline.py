import logging
from datetime import datetime, timedelta
from src.models.signal import TradeSignal

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

    async def process_signal(self, signal: dict, spy_vwap_ratio: float):
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
        vwap_warning = None
        if direction == "Long" and vwap_ratio > 1.025:
            vwap_warning = f"⚠️ Extended — chasing risk (+{(vwap_ratio - 1)*100:.1f}% above VWAP)"
            log.info(f"⚠️ {ticker} Call: Price overextended ({vwap_ratio:.4f}). Tagging alert, not blocking.")
        elif direction == "Short" and vwap_ratio < 0.975:
            vwap_warning = f"⚠️ Extended — chasing risk ({(vwap_ratio - 1)*100:.1f}% below VWAP)"
            log.info(f"⚠️ {ticker} Put: Price overextended ({vwap_ratio:.4f}). Tagging alert, not blocking.")

        # 2. AI Sentiment Check (Blind)
        headlines = await self.news_fetcher.get_headlines(ticker)
        sent_score, catalyst, is_whale, sent_conf = await self.sentiment_analyzer.score_headlines(ticker, headlines)

        if direction == "Short" and sent_score > 0.65:
            log.info(f"🚫 BLOCKING {ticker} Short: Strong positive news ({sent_score:.2f}) contradicts short setup.")
            return
        elif direction == "Long" and sent_score < 0.35:
            log.info(f"🚫 BLOCKING {ticker} Long: Strong negative news ({sent_score:.2f}) contradicts long setup.")
            return

        # 3. XGBoost Sentinel Check → Conviction Tier (never a silent kill)
        features = {
            "entry_time_minute": signal["timestamp"].hour * 60 + signal["timestamp"].minute,
            "entry_volume": signal.get("volume", 100000),
            "vwap_ratio": signal.get("vwap_ratio", 1.0),
            "ema9_ratio": signal.get("ema9_ratio", 1.0),
            "ema_trend": signal.get("ema_trend_bullish", 1),
            "ema_trend_5m": signal.get("ema_trend_5m_bullish", 1),
            "spy_correlation": spy_vwap_ratio,
            "hod_ratio": signal.get("hod_ratio", 1.0),
            "lod_ratio": signal.get("lod_ratio", 1.0)
        }
        xgb_result = self.xgb_model.validate_setup(features)
        win_prob = xgb_result['win_prob']
        verdict = xgb_result['verdict']

        # Conviction tiers based on model output
        if verdict == "CONCORDANT" and win_prob >= 0.75:
            conviction = "🟢 HIGH"
        elif verdict == "CONCORDANT" and win_prob >= 0.55:
            conviction = "🟡 MEDIUM"
        else:
            conviction = "🔴 LOW"

        log.info(f"[{ticker}] AI Filters: Sentiment={sent_score:.2f}, XGB={verdict} ({win_prob:.2f}), Conviction={conviction}")
        log.info(f"✅ {ticker} {direction} Conviction: {conviction}. Dispatching Alert.")

        # Extract natively computed SL/TP if they exist (e.g., from Donchian signals)
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")

        if stop_loss is not None and take_profit is not None:
            risk_levels = self.risk_manager.add_position(
                ticker, signal["entry_price"], initial_atr=0.0, direction=signal["direction"],
                stop_loss=stop_loss, take_profit=take_profit
            )
        else:
            # ORB Signals: attempt to fetch cached real ATR, otherwise fallback with warning
            cached_atr = None
            if self.donchian_strategy and hasattr(self.donchian_strategy, "atr_cache"):
                cached_atr = self.donchian_strategy.atr_cache.get(ticker)
            
            if cached_atr and not pd.isna(cached_atr):
                risk_levels = self.risk_manager.add_position(ticker, signal["entry_price"], initial_atr=cached_atr, direction=signal["direction"])
            else:
                import logging
                logging.getLogger("rallyhunter.pipeline").warning(f"[{ticker}] No cached daily ATR available. Falling back to default ATR=1.5")
                risk_levels = self.risk_manager.add_position(ticker, signal["entry_price"], initial_atr=1.5, direction=signal["direction"])

        # 4. Options Pricing Check (Targeting ~30 days out expiration for short-term breakouts)
        now = datetime.now()
        target_date = now + timedelta(days=30)
        days_to_friday = (4 - target_date.weekday()) % 7
        target_expiration = target_date + timedelta(days=days_to_friday)
        expiration = target_expiration.strftime("%Y-%m-%d")

        target_strike = float(round(signal["entry_price"]))
        if signal["direction"].upper() == "LONG":
            target_strike = float(round(signal["entry_price"] * 1.025)) # 2.5% OTM for balanced cost/leverage
        else:
            target_strike = float(round(signal["entry_price"] * 0.975)) # 2.5% OTM for balanced cost/leverage

        opt_type = "call" if signal["direction"].upper() == "LONG" else "put"
        pricing_data = await self.options_pricer.find_optimal_contract(
            ticker=ticker,
            expiration=expiration,
            target_strike=target_strike,
            option_type=opt_type,
            take_profit=risk_levels["take_profit"]
        )
        
        # Pull the actual chosen strike, which might differ from target_strike due to affordability/liquidity gates
        actual_strike = pricing_data.get("target_strike", target_strike)

        if direction == "Long" and pricing_data.get("flow_bias") == "BEARISH PUT FLOW" and pricing_data.get("put_dollar_flow", 0) > pricing_data.get("call_dollar_flow", 0) * 2.5:
            log.warning(f"[{ticker}] Long alert suppressed: Heavy Bearish Put Flow.")
            return
        elif direction == "Short" and pricing_data.get("flow_bias") == "BULLISH CALL FLOW" and pricing_data.get("call_dollar_flow", 0) > pricing_data.get("put_dollar_flow", 0) * 2.5:
            log.warning(f"[{ticker}] Short alert suppressed: Heavy Bullish Call Flow.")
            return

        # Verdict handling
        strategy_suggestion = None
        iv_value = pricing_data.get("iv", 0.0)
        verdict = pricing_data.get("verdict", "")
        
        if verdict == "POOR VALUE":
            log.warning(f"[{ticker}] Alert suppressed: {pricing_data['reason']}")
            return
        elif verdict == "UNTRADEABLE AT SIZE":
            strategy_suggestion = f"⚠️ Setup valid, not tradeable at your size. {pricing_data['reason']}"
            log.info(f"[{ticker}] Untradeable at size. Tagging alert, not blocking.")
        elif verdict == "OVERPRICED - HIGH IV" or iv_value > 1.20:
            iv_pct = iv_value * 100
            strategy_suggestion = (
                f"💡 IV is rich ({iv_pct:.0f}%) — premium pricing in continued movement. "
                f"Consider a debit spread (buy ${actual_strike} / sell ${actual_strike + 5 if opt_type == 'call' else actual_strike - 5}) "
                f"to cap vega risk instead of a naked long."
            )
            log.info(f"[{ticker}] High IV ({iv_pct:.0f}%). Tagging with spread suggestion, not blocking.")

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
            context_score=signal.get("tf_confluence", "Intraday ORB"),
            catalyst=catalyst,
            vwap_ratio=signal.get("vwap_ratio", 1.0),
            volume=signal.get("volume", 0.0),
            warning_tag=vwap_warning,
            strategy_suggestion=strategy_suggestion,
            stop_loss=risk_levels["stop_loss"],
            take_profit=risk_levels["take_profit"],
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
            expiry=expiration,
            intraday_expiry=expiration,
            target_strike=actual_strike,
            intraday_strike=actual_strike,
            option_type=opt_type.upper(),
            intraday_option_type=opt_type.upper(),
            pricing_verdict=pricing_data.get("verdict", ""),
            pricing_reason=pricing_data.get("reason", "")
        )

        await self.alerts.dispatch_high_conviction(trade_signal)
