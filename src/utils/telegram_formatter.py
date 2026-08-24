from src.models.signal import TradeSignal

def format_telegram_alert(signal: TradeSignal) -> str:

    action_label = "LONG (Calls)" if signal.signal_direction.upper() == "LONG" else "SHORT (Puts)"
    
    header_map = {
        "🔄 FAKEOUT BREAKDOWN CALL REVERSAL": "🔄 <b>FAKEOUT BREAKDOWN CALL REVERSAL (VWAP RECLAIM)</b>",
        "🔄 FAKEOUT BREAKOUT PUT REVERSAL":  "🔄 <b>FAKEOUT BREAKOUT PUT REVERSAL (VWAP REJECT)</b>",
        "🔥 DUAL ORB+CRB CALL BREAKOUT":      "🔥 <b>DUAL ORB+CRB INTRADAY CALL BREAKOUT</b>",
        "🔥 DUAL ORB+CRB PUT BREAKDOWN":     "🔥 <b>DUAL ORB+CRB INTRADAY PUT BREAKDOWN</b>",
        "Closing Range Call Breakout (CRB)":  "🟢 <b>CLOSING RANGE INTRADAY BREAKOUT (CRB CALLS)</b>",
        "Closing Range Put Breakdown (CRB)": "🔴 <b>CLOSING RANGE INTRADAY BREAKDOWN (CRB PUTS)</b>",
        "Opening Range Call Breakout (ORB)":  "🟢 <b>OPENING RANGE INTRADAY BREAKOUT (ORB CALLS)</b>",
        "Opening Range Put Breakdown (ORB)": "🔴 <b>OPENING RANGE INTRADAY BREAKDOWN (ORB PUTS)</b>",
        "15m ORB Breakout + 5m Trend":        "🟢 <b>INTRADAY CALL BREAKOUT (15m ORB + 5m TREND)</b>",
        "Bullish 5m+1m Trend Continuation":   "🟢 <b>BULLISH TREND CONTINUATION (CALLS)</b>",
        "15m ORB Breakdown + 5m Trend":       "🔴 <b>INTRADAY PUT BREAKDOWN (15m ORB + 5m TREND)</b>",
        "Bearish 5m+1m Trend Breakdown":      "🔴 <b>BEARISH TREND BREAKDOWN (PUTS)</b>",
        "Volume Breakout":                    "💥 <b>VOLUME BREAKOUT</b>",
        "Trend Breakout":                     "📈 <b>TREND BREAKOUT</b>",
        "Volume Breakdown":                   "📉 <b>VOLUME BREAKDOWN</b>",
        "Trend Breakdown":                    "📉 <b>TREND BREAKDOWN</b>",
        "Intraday Volume Breakout":           "🔺 <b>INTRADAY VOLUME BREAKOUT</b>",
        "Intraday Trend Breakout":            "🔺 <b>INTRADAY TREND BREAKOUT</b>",
        "Intraday Volume Breakdown":          "🔻 <b>INTRADAY VOLUME BREAKDOWN</b>",
        "Intraday Trend Breakdown":           "🔻 <b>INTRADAY TREND BREAKDOWN</b>",
        "Pullback Retest":                    "🟢 <b>PULLBACK RETEST (SUPPORT HOLD)</b>",
        "Pullback Retest Short":              "🔴 <b>BEARISH PULLBACK RETEST</b>",
        "Daily Mean Reversion":               "🔄 <b>DAILY MEAN REVERSION (BOUNCE)</b>"
    }
    
    strategy_header = header_map.get(signal.strategy_type)
    if not strategy_header:
        if signal.strategy_type.startswith("Failed "):
            parent_strategy = signal.strategy_type.replace("Failed ", "")
            opt_type = "CALLS" if signal.signal_direction.upper() == "LONG" else "PUTS"
            strategy_header = f"🔄 <b>{parent_strategy.upper()} FAKEOUT REVERSAL ({opt_type})</b>"
        else:
            strategy_header = "⚡ <b>SWING OPPORTUNITY</b>"
    
    ticker_code = f"<code>{signal.ticker}</code>"
    
    # Options target line (only if we have real data)
    options_lines = ""
    timeframe = signal.timeframe_target.upper()
    if timeframe == "SWING" and signal.target_strike:
        option_details = f"{signal.expiry} ${signal.target_strike} {signal.option_type}"
        options_lines = f"🎯 <b>Target (Swing):</b> <code>{option_details}</code>\n"
    elif timeframe == "INTRADAY" and signal.intraday_strike:
        option_details = f"{signal.intraday_expiry} ${signal.intraday_strike} {signal.intraday_option_type}"
        options_lines = f"⚡ <b>Target (Intraday):</b> <code>{option_details}</code>\n"
        if signal.delta > 0:
            iv_str = f"{signal.iv*100:.1f}%" if signal.iv > 0 else "N/A"
            options_lines += "⚖️ <b>Greeks & Premium:</b>\n"
            options_lines += f"   • Premium: <code>${signal.option_ask:.2f}</code>\n"
            options_lines += f"   • Delta: <code>{signal.delta:.2f}</code>\n"
            options_lines += f"   • Theta: <code>{signal.theta:.4f}</code>\n"
            options_lines += f"   • IV: <code>{iv_str}</code>\n"
            
            verdict_color = "✅" if signal.pricing_verdict == "FAIR VALUE" else "⚠️"
            options_lines += f"{verdict_color} <b>Pricing:</b> {signal.pricing_verdict or 'UNKNOWN'} - <i>{signal.pricing_reason}</i>\n"
        else:
            options_lines += "⚠️ <b>Warning:</b> Short-term weekly option. Exit if target not hit within 60 mins.\n"
    
    whale_line = "🐳 <b>Whale Flow:</b> <code>$100k+ Detected</code>\n" if signal.is_whale else ""
    
    # VWAP distance (real computed value)
    vwap_pct = (signal.vwap_ratio - 1.0) * 100
    vwap_label = f"+{vwap_pct:.2f}% above" if vwap_pct >= 0 else f"{vwap_pct:.2f}% below"
    
    warning_line = f"{signal.warning_tag}\n" if signal.warning_tag else ""
    earnings_line = f"⚠️ <b>Earnings Risk:</b> <code>{signal.earnings_date}</code> falls within this option's holding window\n" if getattr(signal, "earnings_risk", False) else ""
    
    flow_line = f"💵 <b>Options Flow:</b> <code>{signal.flow_bias}</code> (${signal.call_dollar_flow:,.0f} Calls / ${signal.put_dollar_flow:,.0f} Puts)\n"
    
    msg = (
        f"{strategy_header}\n"
        f"Asset: {ticker_code} ({action_label})\n"
        f"✅ <b>STATUS:</b> CONFIRMED\n"
        f"{warning_line}\n"
        f"{earnings_line}"
        f"{options_lines}"
        f"🧠 <b>Confluence:</b> <i>{signal.context_score}</i>\n"
        f"📊 <b>Conviction:</b> {signal.conviction} (<code>{int(signal.win_probability * 100)}%</code>)\n\n"
        f"📈 <b>Market Context:</b>\n"
        f"• Price: <code>${signal.price:.2f}</code>\n"
        f"• VWAP: <code>{vwap_label}</code>\n"
        f"• Catalyst: <i>{signal.catalyst}</i>\n"
        f"{flow_line}"
        f"{whale_line}\n"
        f"🛡️ <b>EXIT MATRIX (One-Tap Copy):</b>\n"
        f"• Stock TP: <code>${signal.take_profit:.2f}</code> | SL: <code>${signal.stop_loss:.2f}</code>\n"
    )
    
    if signal.option_ask > 0:
        opt_tp = round(signal.option_ask * (1 + signal.option_tp_pct), 2)
        opt_sl = round(signal.option_ask * (1 + signal.option_sl_pct), 2)
        midpoint = round((signal.option_ask + (signal.option_bid or signal.option_ask)) / 2, 2)
        msg += (
            f"• Limit Entry (Mid): <code>${midpoint:.2f}</code>\n"
            f"• Option TP: <code>${opt_tp:.2f}</code> (+{int(signal.option_tp_pct*100)}%) "
            f"| SL: <code>${opt_sl:.2f}</code> ({int(signal.option_sl_pct*100)}%)\n"
            f"  <i>Bid/Ask: ${signal.option_bid or 0.0:.2f} / ${signal.option_ask:.2f} | Est. Cost: ${int(midpoint*100)}/contract</i>"
        )
    
    if signal.strategy_suggestion:
        msg += f"\n\n{signal.strategy_suggestion}"
        
    if getattr(signal, "gex_confidence", "STANDARD") == "HIGH":
        msg += "\n\n📌 <b>Post-earnings gamma compression window</b> — strike selection has stronger historical backing here"
        
    return msg
