import os
import logging
import aiohttp
import pandas as pd
from typing import Optional, Any
from src.data.dynamic_scanner import CANDIDATE_POOL, MEGA_CAP_TICKERS

log = logging.getLogger(__name__)

class HourlySnapshotEngine:
    """
    Generates hourly market breadth and momentum snapshots across all watched tickers.
    Queries bulk quotes from Tradier in a single call to protect API limits.
    """
    def __init__(self, tradier_token: Optional[str] = None):
        self.token = tradier_token or os.getenv("TRADIER_ACCESS_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }

    async def generate_market_snapshot(self, session: Optional[aiohttp.ClientSession] = None, alert_gateway: Optional[Any] = None) -> Optional[str]:
        """
        Pulls real-time bulk quotes for the watchlist universe and builds a formatted HTML Telegram card.
        Optionally appends live catalyst reliability stats if available.
        """
        if not self.token:
            log.warning("No Tradier token configured for hourly snapshot.")
            return None

        symbols_str = ",".join(CANDIDATE_POOL)
        url = f"https://api.tradier.com/v1/markets/quotes?symbols={symbols_str}&greeks=false"

        try:
            async def do_fetch(s):
                async with s.get(url, headers=self.headers, timeout=10.0) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    log.warning(f"Tradier quotes status {resp.status} during hourly snapshot.")
                    return None

            if session:
                data = await do_fetch(session)
            else:
                async with aiohttp.ClientSession() as s:
                    data = await do_fetch(s)

            if not data:
                return None

            quotes = data.get("quotes", {}).get("quote", [])
            if not isinstance(quotes, list):
                quotes = [quotes]

            if not quotes:
                return None

            # Process ticker data
            mega_caps = []
            mid_caps = []
            indexes = {}
            total_up = 0
            total_down = 0
            total_dollar_vol = 0.0

            for q in quotes:
                sym = q.get("symbol", "").upper()
                last = float(q.get("last") or q.get("close") or 0.0)
                prevclose = float(q.get("prevclose") or last or 1.0)
                vol = float(q.get("volume") or 0.0)
                chg_pct = ((last - prevclose) / prevclose * 100.0) if prevclose > 0 else 0.0
                dollar_vol = vol * last

                if last <= 0:
                    continue

                total_dollar_vol += dollar_vol
                if chg_pct >= 0:
                    total_up += 1
                else:
                    total_down += 1

                item = {
                    "symbol": sym,
                    "price": last,
                    "change_pct": chg_pct,
                    "volume": vol,
                    "dollar_vol": dollar_vol
                }

                if sym in ("SPY", "QQQ", "IWM", "GLD"):
                    indexes[sym] = item
                elif sym in MEGA_CAP_TICKERS:
                    mega_caps.append(item)
                else:
                    mid_caps.append(item)

            # Sort gainers
            mega_caps.sort(key=lambda x: x["change_pct"], reverse=True)
            mid_caps.sort(key=lambda x: x["change_pct"], reverse=True)

            # Breadth calculation
            total_counted = total_up + total_down
            breadth_pct = (total_up / total_counted * 100.0) if total_counted > 0 else 50.0
            breadth_mood = "🟢 BULLISH BREADTH" if breadth_pct >= 60 else ("🔴 BEARISH BREADTH" if breadth_pct <= 40 else "🟡 BALANCED")

            now_est = pd.Timestamp.now(tz="US/Eastern").strftime("%I:%M %p EST")

            # Format indices line
            spy_info = indexes.get("SPY", {"price": 0, "change_pct": 0})
            qqq_info = indexes.get("QQQ", {"price": 0, "change_pct": 0})
            iwm_info = indexes.get("IWM", {"price": 0, "change_pct": 0})

            spy_emoji = "🟢" if spy_info["change_pct"] >= 0 else "🔴"
            qqq_emoji = "🟢" if qqq_info["change_pct"] >= 0 else "🔴"
            iwm_emoji = "🟢" if iwm_info["change_pct"] >= 0 else "🔴"

            # Top 3 Mega-Cap leaders
            mega_lines = ""
            for m in mega_caps[:3]:
                e = "🟢" if m["change_pct"] >= 0 else "🔴"
                mega_lines += f"  • {e} <b>{m['symbol']}</b>: <code>${m['price']:.2f}</code> ({m['change_pct']:+.2f}%)\n"

            # Top 3 Mid-Cap runners
            mid_lines = ""
            for m in mid_caps[:3]:
                e = "🟢" if m["change_pct"] >= 0 else "🔴"
                mid_lines += f"  • {e} <b>{m['symbol']}</b>: <code>${m['price']:.2f}</code> ({m['change_pct']:+.2f}%)\n"

            # Optional Catalyst Reliability section (if >=3 decided trades exist)
            catalyst_section = ""
            if alert_gateway and hasattr(alert_gateway, "get_catalyst_report"):
                report = alert_gateway.get_catalyst_report()
                if report:
                    catalyst_section = f"\n\n{report}"

            msg = (
                f"⏱️ <b>HOURLY MARKET PULSE SNAPSHOT</b> ({now_est})\n\n"
                f"📊 <b>Core Index Benchmarks:</b>\n"
                f"  • {spy_emoji} <b>SPY:</b> <code>${spy_info['price']:.2f}</code> ({spy_info['change_pct']:+.2f}%)\n"
                f"  • {qqq_emoji} <b>QQQ:</b> <code>${qqq_info['price']:.2f}</code> ({qqq_info['change_pct']:+.2f}%)\n"
                f"  • {iwm_emoji} <b>IWM:</b> <code>${iwm_info['price']:.2f}</code> ({iwm_info['change_pct']:+.2f}%)\n\n"
                f"🌡️ <b>Market Breadth:</b> {breadth_mood} (<code>{breadth_pct:.0f}% Advancing</code>)\n"
                f"💵 <b>Watchlist Dollar Vol:</b> <code>${total_dollar_vol / 1e9:.2f}B</code>\n\n"
                f"🏢 <b>Top Mega-Cap Leaders:</b>\n"
                f"{mega_lines}\n"
                f"🚀 <b>Top High-Beta Mid-Cap Movers:</b>\n"
                f"{mid_lines}"
                f"{catalyst_section}\n\n"
                f"🎯 <i>Monitoring 4-Timeframe & 0.75 Delta ITM breakout entries...</i>"
            )

            return msg

        except Exception as e:
            log.error(f"Error generating hourly market snapshot: {e}")
            return None
