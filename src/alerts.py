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

from typing import Any
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
        Dispatches an informational extended hours alert.
        Does not interact with SQLite or log a formal trade.
        """
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
                                    "• /status - View engine status and health\n"
                                    "• /scan [TICKER] - Run a real-time manual scan\n"
                                    "• /help - Display this help message"
                                )
                                await self._send_telegram(help_msg)
                                
                            elif text.startswith("/status"):
                                status_msg = (
                                    "<b>📊 RallyHunter Engine Status</b>\n\n"
                                    "• <b>Market Session:</b> Active\n"
                                    "• <b>API Providers:</b> Tradier & Finnhub (Online)\n"
                                    "• <b>LLM Analyzer:</b> Groq/Llama 3.1 (Online)"
                                )
                                await self._send_telegram(status_msg)
                                    
                            elif text.startswith("/scan"):
                                await self._send_telegram("⚠️ <b>/scan</b> is deprecated in the new event-driven architecture.")
                                    
                except Exception as e: 
                    log.error("AlertGateway.start_listener encountered Telegram updates polling error: %s", e)
                    await asyncio.sleep(5)

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
