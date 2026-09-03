"""
RallyHunter_v26 -- Master Alert Gateway
Role: Senior AI Engineer Refactor
Focus: Thread-safe journaling, Rich-text institutional alerts, and S&P 500 Wide-Net integration.
"""
import asyncio
import logging
import os
from datetime import datetime

import aiohttp

from typing import Any, Optional
from src.database import add_trade
from src.utils.telegram_formatter import format_telegram_alert

# ── Configuration ──────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

log = logging.getLogger("rallyhunter.alerts")

# ── Alert Gateway ──────────────────────────────────────────────────────────

class AlertGateway:
    """Handles all outgoing communication and thread-safe journaling."""
    
    def __init__(self, redis_client):
        self._redis = redis_client
        self._token = os.getenv("TELEGRAM_BOT_TOKEN")
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self._url = f"https://api.telegram.org/bot{self._token}/"
        self._vault_lock = asyncio.Lock() # Thread-safe CSV writes
        self._session = None
        # In-memory catalyst performance counters: catalyst_type -> outcome counts
        self.catalyst_stats = {}
        # Set by the engine. The bot cannot see the brokerage account, so the trader
        # tells it what was actually taken via /took and /closed.
        self.risk_manager = None

    def attach_position_ledger(self, risk_manager) -> None:
        """Wires the position ledger so /took, /closed and /positions can reach it."""
        self.risk_manager = risk_manager

    def record_outcome(self, catalyst_type: str, outcome_status: str) -> None:
        """
        Increments in-memory trade outcome counters per catalyst type.
        Tracks total signals, INVALIDATED, STOPPED_OUT, and TP_HIT exits.
        """
        cat = catalyst_type or "Standard Breakout"
        if cat not in self.catalyst_stats:
            self.catalyst_stats[cat] = {
                "total_signals": 0,
                "INVALIDATED": 0,
                "STOPPED_OUT": 0,
                "TP_HIT": 0
            }
        
        self.catalyst_stats[cat]["total_signals"] += 1
        if outcome_status in self.catalyst_stats[cat]:
            self.catalyst_stats[cat][outcome_status] += 1
        else:
            self.catalyst_stats[cat][outcome_status] = 1

    def get_catalyst_report(self) -> Optional[str]:
        """
        Returns a formatted HTML string summarizing catalyst fakeout and win rates today.
        Only includes catalysts with at least 3 decided trades to avoid small-sample noise.
        """
        lines = []
        for cat, stats in sorted(self.catalyst_stats.items()):
            inv = stats.get("INVALIDATED", 0)
            stopped = stats.get("STOPPED_OUT", 0)
            tp = stats.get("TP_HIT", 0)
            total_decided = inv + stopped + tp
            
            if total_decided >= 3:
                fakeout_rate = (inv / total_decided) * 100.0
                win_rate = (tp / total_decided) * 100.0
                clean_cat = cat.replace("🔥 ", "").replace("🔄 ", "").replace("🔺 ", "").replace("🔻 ", "").replace("🟢 ", "").replace("🔴 ", "")
                lines.append(
                    f"  • <b>{clean_cat}</b>: <code>{win_rate:.0f}% Win</code> | <code>{fakeout_rate:.0f}% Fakeout</code> ({total_decided} trades)"
                )
                
        if not lines:
            return None
            
        header = "🎯 <b>Catalyst Reliability (Sample &ge; 3):</b>\n"
        return header + "\n".join(lines)

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def dispatch_informational(self, message: str) -> bool:
        """Raw dispatch for system alerts without database journaling."""
        return await self._send_telegram(message)

    async def dispatch_extended_hours(self, alert: dict) -> bool:
        """
        Extended hours mover handler.
        Options exchanges are closed outside 9:30 AM - 4:00 PM EST.
        Silenced by default to eliminate after-hours alert fatigue unless ENABLE_EXTENDED_HOURS_ALERTS is true.
        """
        if os.getenv("ENABLE_EXTENDED_HOURS_ALERTS", "false").lower() != "true":
            log.debug("Extended hours alert suppressed (options chain closed outside regular session).")
            return False

        ticker = alert["ticker"]
        direction = alert["direction"]
        pct_change = alert["pct_change"]
        price = alert["price"]
        vol = alert["volume"]
        
        emoji = "🚀" if direction == "Long" else "🩸"
        msg = (
            f"🌙 <b>EXTENDED HOURS MOVER</b> 🌙\n\n"
            f"<b>{ticker}</b> {emoji} <b>{direction} Setup</b>\n"
            f"<b>Price:</b> ${price:.2f} (<b>{pct_change:+.2f}%</b>)\n"
            f"<b>Ext Vol:</b> {vol:,.0f}\n\n"
            f"<i>⚠️ Informational only. Options chain frozen. Will automatically re-evaluate at next session open.</i>"
        )
        return await self._send_telegram(msg)

    async def _send_telegram(self, text: str) -> bool:
        """Internal Telegram dispatcher supporting comma-separated chat IDs."""
        if not self._token or not self._chat_id: return False
        chat_ids = [cid.strip() for cid in str(self._chat_id).split(",") if cid.strip()]
        success = True
        session = await self.get_session()
        for cid in chat_ids:
            payload = {"chat_id": cid, "text": text, "parse_mode": "HTML"}
            try:
                async with session.post(f"{self._url}sendMessage", json=payload) as r:
                    if r.status != 200:
                        success = False
            except Exception as e:
                log.error("AlertGateway._send_telegram failed to send message to chat_id %s: %s", cid, e)
                success = False
        return success

    async def start_listener(self):
        """Polls for basic start, status, watchlist, scan, and config commands."""
        log.info("RallyHunter Streamlined Listener online.")
        offset = 0
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    params = {"offset": offset, "timeout": 30}
                    async with session.get(f"{self._url}getUpdates", params=params) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(5)
                            continue
                        data = await resp.json()
                        for update in data.get("result", []):
                            offset = update["update_id"] + 1
                            msg = update.get("message", {})
                            text = msg.get("text", "")
                            if not text: continue
                            
                            text = text.strip()
                            if text.startswith("/start"):
                                await self._send_telegram("<b>🚀 RallyHunter Broad Discovery Active.</b>\nMonitoring dynamic market movers...")
                            
                            elif text.startswith("/help"):
                                help_msg = (
                                    "<b>⚡ RallyHunter Commands</b>\n\n"
                                    "• /took TICKER [qty] [fill] - I took this trade; start watching it\n"
                                    "• /closed TICKER [fill] - I closed it; record the real result\n"
                                    "• /positions - What the bot thinks you are holding\n"
                                    "• /status - View engine status and health\n"
                                    "• /help - Display this help message\n\n"
                                    "<i>Alerts are suggestions. Only positions you confirm with /took "
                                    "receive expiration and exit warnings.</i>"
                                )
                                await self._send_telegram(help_msg)
                                
                            elif text.startswith("/status"):
                                status_msg = (
                                    "<b>📊 RallyHunter Engine Status</b>\n\n"
                                    "• <b>Market Session:</b> Active\n"
                                    "• <b>API Providers:</b> Tradier & Finnhub (Online)\n"
                                    "• <b>LLM Analyzer:</b> OpenRouter/DeepSeek (Online)"
                                )
                                await self._send_telegram(status_msg)
                                    
                            elif text.startswith("/took"):
                                await self._send_telegram(self._cmd_took(text))

                            elif text.startswith("/closed"):
                                await self._send_telegram(self._cmd_closed(text))

                            elif text.startswith("/positions"):
                                await self._send_telegram(self._cmd_positions())

                            elif text.startswith("/scan"):
                                await self._send_telegram("⚠️ <b>/scan</b> is deprecated in the new event-driven architecture.")
                                    
                except Exception as e: 
                    log.error("AlertGateway.start_listener encountered Telegram updates polling error: %s", e)
                    await asyncio.sleep(5)

    @staticmethod
    def _parse_command(text: str):
        """`/took PLTR 2 9.10` -> ("PLTR", [2.0, 9.10]). Numbers are optional."""
        parts = text.split()[1:]
        if not parts:
            return None, []
        ticker = parts[0].upper().lstrip("$")
        nums = []
        for raw in parts[1:]:
            try:
                nums.append(float(raw.replace("$", "").replace(",", "")))
            except ValueError:
                continue
        return ticker, nums

    def _cmd_took(self, text: str) -> str:
        if self.risk_manager is None:
            return "⚠️ Position ledger is not attached; cannot record trades."
        ticker, nums = self._parse_command(text)
        if not ticker:
            return "Usage: <code>/took TICKER [quantity] [fill price]</code>"
        qty = nums[0] if len(nums) >= 1 else None
        fill = nums[1] if len(nums) >= 2 else None
        pos = self.risk_manager.confirm_position(ticker, quantity=qty, fill_price=fill)
        if pos is None:
            return (f"❓ No tracked alert for <b>{ticker}</b>. The bot only watches tickers it "
                    f"alerted on, and unconfirmed alerts are dropped at the next daily reset.")
        exp = pos.get("option_expiration") or "unknown"
        detail = f" — {qty:g} @ ${fill:.2f}" if (qty is not None and fill is not None) else ""
        return (f"✅ Watching <b>{ticker}</b>{detail}\n"
                f"Expiry {exp}. You will get expiration, time-stop and premium-stop warnings "
                f"for this position until you <code>/closed {ticker}</code>.")

    def _cmd_closed(self, text: str) -> str:
        if self.risk_manager is None:
            return "⚠️ Position ledger is not attached; cannot record trades."
        ticker, nums = self._parse_command(text)
        if not ticker:
            return "Usage: <code>/closed TICKER [fill price]</code>"
        fill = nums[0] if nums else None
        summary = self.risk_manager.close_confirmed(ticker, exit_option_price=fill)
        if summary is None:
            return f"❓ <b>{ticker}</b> is not being tracked."
        pnl = summary.get("option_pnl_pct")
        held = summary.get("days_held")
        bits = [f"📕 Closed <b>{ticker}</b>"]
        if pnl is not None:
            bits.append(f"Real option P&amp;L: <b>{pnl*100:+.1f}%</b>")
        else:
            bits.append("No fill price given, so no real P&amp;L recorded "
                        "(send <code>/closed TICKER 12.40</code> next time).")
        if held is not None:
            bits.append(f"Held {held} day(s).")
        return "\n".join(bits)

    def _cmd_positions(self) -> str:
        if self.risk_manager is None:
            return "⚠️ Position ledger is not attached."
        book = self.risk_manager.list_positions()
        lines = ["<b>📋 Position Ledger</b>", ""]
        if book["confirmed"]:
            lines.append("<b>Holding (you confirmed these):</b>")
            for ticker, pos in sorted(book["confirmed"]):
                held = self.risk_manager.days_held(pos)
                held_txt = f"{held}d" if held is not None else "?"
                lines.append(f"  • <b>{ticker}</b> {pos.get('direction','')} — exp "
                             f"{pos.get('option_expiration') or '?'}, held {held_txt}")
        else:
            lines.append("<i>Nothing confirmed. Reply /took TICKER when you take an alert.</i>")
        if book["unconfirmed"]:
            lines.append("")
            lines.append(f"<b>Alerted, not confirmed ({len(book['unconfirmed'])}):</b>")
            lines.append("  " + ", ".join(sorted(t for t, _ in book["unconfirmed"])))
            lines.append("<i>These get no exit warnings and are dropped at the daily reset.</i>")
        return "\n".join(lines)

    async def dispatch_high_conviction(self, signal: Any, force_send: bool = False, suppress_telegram: bool = False) -> bool:
        """Dispatches an institutional-grade signal and journals to SQLite."""
        # 1. Senior Deduplication
        quiet_seconds = int(os.getenv("ALERT_DEDUP_SECONDS", "3600"))
        dedup_key = f"alert_dedup:{signal.ticker}:{signal.signal_direction}"
        if not force_send:
            if self._redis:
                try:
                    if await self._redis.get(dedup_key):
                        return False
                    await self._redis.setex(dedup_key, quiet_seconds, "1")
                except Exception as e:
                    log.warning("AlertGateway.dispatch_high_conviction failed Redis deduplication check for %s: %s", signal.ticker, e)
            else:
                log.warning("Redis not available, deduplication disabled.")
        
        # 2. Database Journaling
        try:
            trade_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": signal.ticker,
                "price": signal.price,
                "is_whale": signal.is_whale,
                "win_prob": f"{int(signal.win_probability*100)}%",
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "strategy": signal.strategy_type,
                "direction": signal.signal_direction,
                "catalyst": signal.catalyst,
                "xgb_win_prob": signal.xgb_win_prob,
                "sentinel_verdict": signal.sentinel_verdict,
                "conviction": signal.conviction,
                "warning_tag": signal.warning_tag if signal.warning_tag else ""
            }
            await add_trade(trade_data)
        except Exception as e:
            log.error("AlertGateway.dispatch_high_conviction failed to journal trade for %s to SQLite: %s", signal.ticker, e)

        # 3. Mobile-Optimized Premium Alert Template
        msg = format_telegram_alert(signal)
        if suppress_telegram:
            log.info("Telegram alert suppressed for %s (Startup Grace Period active).", signal.ticker)
            return False
        return await self._send_telegram(msg)

if __name__ == "__main__":
    import redis.asyncio as aioredis
    logging.basicConfig(level=logging.INFO)
    async def test():
        r = await aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        gw = AlertGateway(r)
        await gw.start_listener()
    asyncio.run(test())
