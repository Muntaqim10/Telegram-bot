import logging
from datetime import datetime, timedelta
from src.models.signal import TradeSignal

log = logging.getLogger("rallyhunter.pipeline")

class SignalPipeline:
    def __init__(self, news_fetcher, sentiment_analyzer, xgb_model, risk_manager, options_pricer, alerts_gateway):
        self.news_fetcher = news_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        self.xgb_model = xgb_model
        self.risk_manager = risk_manager
        self.options_pricer = options_pricer
        self.alerts = alerts_gateway

    async def process_signal(self, signal: dict, spy_vwap_ratio: float):
        """Passes the raw signal through AI filters before alerting."""
        ticker = signal["ticker"]
        direction = signal["direction"]

        # 1. Market Bias Tracking
        if direction == "Long":
            log.info(f"🟢 CALL BIAS ACTIVE: {ticker} Long breakout setup evaluated without SPY block.")
        else:
            log.info(f"🔴 {ticker} Short breakdown setup evaluated with risk management.")

        # 1.5 VWAP Overextension Guard (+/- 2.5% from VWAP)
        vwap_ratio = signal.get("vwap_ratio", 1.0)
        if direction == "Long" and vwap_ratio > 1.025:
            log.info(f"🚫 BLOCKING {ticker} Call: Price is overextended ({vwap_ratio:.4f} > +2.5% above VWAP). High pullback risk.")
            return
        elif direction == "Short" and vwap_ratio < 0.975:
            log.info(f"🚫 BLOCKING {ticker} Put: Price is overextended ({vwap_ratio:.4f} < -2.5% below VWAP). High bounce risk.")
            return

        # 2. AI Sentiment Check (Blind)
        headlines = await self.news_fetcher.get_headlines(ticker)
        sent_score, catalyst, is_whale, sent_conf = await self.sentiment_analyzer.score_headlines(ticker, headlines)

        if direction == "Short" and sent_score > 0.65:
            log.info(f"🚫 BLOCKING {ticker} Short: Strong positive news ({sent_score:.2f}) contradicts short setup.")
            return
        elif direction == "Long" and sent_score < 0.35:
            log.info(f"🚫 BLOCKING {ticker} Long: Strong negative news ({sent_score:.2f}) contradicts long setup.")
            return

        # 3. XGBoost Sentinel Check (Fakeout Filter)
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
        
        log.info(f"[{ticker}] AI Filters: Sentiment={sent_score:.2f}, XGB={xgb_result['verdict']} ({xgb_result['win_prob']:.2f})")

        # STRICT ENFORCEMENT: Fakeout Filter
        if xgb_result['verdict'] != "CONCORDANT" or xgb_result['win_prob'] < 0.75:
            log.info(f"❌ {ticker} Rejected by Fakeout Filter. Prob: {xgb_result['win_prob']}, Verdict: {xgb_result['verdict']}")
            return

        log.info(f"✅ {ticker} {direction} Passed AI Filters! Dispatching Alert.")

        # Register in Risk Manager
        self.risk_manager.add_position(ticker, signal["entry_price"], initial_atr=1.5, direction=signal["direction"])

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
        pricing_data = await self.options_pricer.evaluate_contract(
            ticker=ticker,
            expiration=expiration,
            strike=target_strike,
            option_type=opt_type
        )

        gex_strike = pricing_data.get("gex_target_strike")
        if gex_strike and gex_strike != target_strike:
            log.info(f"[{ticker}] GEX Target Strike Identified: ${gex_strike:.2f}")
            target_strike = gex_strike

        if direction == "Long" and pricing_data.get("flow_bias") == "BEARISH PUT FLOW" and pricing_data.get("put_dollar_flow", 0) > pricing_data.get("call_dollar_flow", 0) * 2.5:
            log.warning(f"[{ticker}] Long alert suppressed: Heavy Bearish Put Flow.")
            return
        elif direction == "Short" and pricing_data.get("flow_bias") == "BULLISH CALL FLOW" and pricing_data.get("call_dollar_flow", 0) > pricing_data.get("put_dollar_flow", 0) * 2.5:
            log.warning(f"[{ticker}] Short alert suppressed: Heavy Bullish Call Flow.")
            return

        if pricing_data["verdict"] in ["POOR VALUE", "OVERPRICED - HIGH IV"]:
            log.warning(f"[{ticker}] Alert suppressed by Options Pricer: {pricing_data['reason']}")
            return

        # 5. Build TradeSignal Object
        trade_signal = TradeSignal(
            ticker=ticker,
            price=signal["entry_price"],
            signal_direction=signal["direction"],
            strategy_type=signal["catalyst_type"],
            timestamp=signal["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(signal["timestamp"], "strftime") else str(signal["timestamp"]),
            z_vol=3.5,
            is_whale=is_whale or pricing_data.get("is_whale", False),
            rev_growth=30.0,
            win_probability=xgb_result['win_prob'],
            xgb_win_prob=xgb_result['win_prob'],
            sentinel_verdict=xgb_result['verdict'],
            context_score="Strong Intraday ORB",
            catalyst=catalyst,
            historical_win_rate=80.0,
            rsi=60.0,
            stop_loss=0.0,
            take_profit=0.0,
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
            target_strike=target_strike,
            intraday_strike=target_strike,
            option_type=opt_type.upper(),
            intraday_option_type=opt_type.upper(),
            pricing_verdict=pricing_data.get("verdict", ""),
            pricing_reason=pricing_data.get("reason", "")
        )

        await self.alerts.dispatch_high_conviction(trade_signal)
