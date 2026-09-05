"""Marks and closes the paper trades opened by each alert.

The alerter has produced 523 alerts and zero confirmed positions, so every performance
figure it reports is the trailing stop's simulation. Confirmation cannot fix that when
the trader does not take the alerts, and the backtest cannot either: it prices with
Black-Scholes on realized volatility and tests four setups sharing nothing with the live
alert stream.

Forward paper trading can, because the chain is retrievable in real time. Each alert
opens a position on the contract it actually named, at the ask it was quoted, and this
module closes it at the real bid.

Deliberately pessimistic where it is uncertain:

- entry at the ASK, exit at the BID, so the round trip pays the spread twice, as a real
  one does. The old backtest used a theoretical mid and charged 3% on exit only.
- a contract that cannot be quoted is closed at zero, not skipped. Illiquid strikes are
  a real cost of trading them, and dropping them would quietly select for the ones that
  worked.
- exits are mechanical: the hold ceiling, or expiry. No discretion, no hindsight.
"""
import logging
import os
from datetime import datetime

import aiohttp
import pandas as pd

from src.database import open_paper_positions, settle_paper_trade

log = logging.getLogger("rallyhunter.paper")

QUOTES_URL = "https://api.tradier.com/v1/markets/quotes"

# Paper trades close on the same ceiling live positions use, so the two are comparable.
# Measured on the account's own fills, the median trade is already losing by day 4.
DEFAULT_HOLD_DAYS = 3


def _market_today() -> datetime.date:
    return pd.Timestamp.now(tz="US/Eastern").date()


async def fetch_option_quotes(symbols, session=None, token=None):
    """Current bid/ask for a batch of OCC contracts. Missing symbols are simply absent."""
    token = token or os.getenv("TRADIER_ACCESS_TOKEN")
    if not symbols or not token:
        return {}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    out = {}
    own = session is None
    session = session or aiohttp.ClientSession()
    try:
        for i in range(0, len(symbols), 40):
            chunk = ",".join(symbols[i:i + 40])
            try:
                async with session.get(QUOTES_URL, params={"symbols": chunk},
                                       headers=headers, timeout=10) as r:
                    if r.status != 200:
                        log.warning("Option quote fetch returned %s for %d symbol(s).",
                                    r.status, len(symbols[i:i + 40]))
                        continue
                    data = await r.json()
            except Exception as e:
                log.warning("Option quote fetch failed: %s", e)
                continue
            quotes = (data.get("quotes") or {}).get("quote") or []
            if isinstance(quotes, dict):
                quotes = [quotes]
            for q in quotes:
                if q.get("symbol"):
                    out[q["symbol"]] = {"bid": q.get("bid"), "ask": q.get("ask")}
    finally:
        if own:
            await session.close()
    return out


def _days_held(opened_at: str) -> int:
    try:
        return (_market_today() - datetime.strptime(
            opened_at[:10], "%Y-%m-%d").date()).days
    except Exception:
        return 0


def _exit_reason(trade, days, today):
    """Mechanical exits only. Discretion here would be hindsight."""
    expiry = (trade.get("expiry") or "")[:10]
    if expiry:
        try:
            if datetime.strptime(expiry, "%Y-%m-%d").date() <= today:
                return "EXPIRED"
        except ValueError:
            pass
    if days >= int(os.getenv("PAPER_HOLD_DAYS", DEFAULT_HOLD_DAYS)):
        return "HOLD_CEILING"
    return None


async def sweep(session=None, token=None) -> dict:
    """Mark every open paper trade, closing those that have reached an exit.

    Returns a small summary so the caller can log one line rather than a hundred.
    """
    trades = await open_paper_positions()
    if not trades:
        return {"open": 0, "marked": 0, "closed": 0}

    quotes = await fetch_option_quotes([t["occ_symbol"] for t in trades],
                                       session=session, token=token)
    today = _market_today()
    now = pd.Timestamp.now(tz="US/Eastern").strftime("%Y-%m-%d %H:%M:%S")
    marked = closed = 0

    for t in trades:
        q = quotes.get(t["occ_symbol"]) or {}
        bid = q.get("bid")
        days = _days_held(t["opened_at"])
        reason = _exit_reason(t, days, today)

        if reason is None:
            if bid is not None:
                await settle_paper_trade(t["id"], marked_at=now, mark_bid=bid)
                marked += 1
            continue

        # Closing. An unquotable contract is worth what you could sell it for: nothing.
        exit_bid = bid if bid is not None else 0.0
        if bid is None:
            log.info("[%s] %s could not be quoted at close; settled at 0.",
                     t["ticker"], t["occ_symbol"])
        entry = float(t["entry_ask"] or 0.0)
        pnl = ((exit_bid - entry) / entry * 100.0) if entry else None
        await settle_paper_trade(t["id"], closed_at=now, exit_bid=exit_bid,
                                 exit_reason=reason, pnl_pct=pnl, days_held=days)
        closed += 1

    return {"open": len(trades), "marked": marked, "closed": closed}
