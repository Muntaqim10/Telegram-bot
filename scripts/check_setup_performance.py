"""Per-setup performance, with an honest sample size.

Retiring a setup is the cheapest P&L change available, and also the easiest to get
wrong. The raw table is seductive: on 2026-09-04 the worst setup showed 81 alerts at
45.7% and -3.29% over three days, which reads like proof. It is not. Those 81 alerts
landed on five distinct days, and alerts fired on the same day move together -- one bad
session supplies dozens of correlated "observations".

So this reports the day count next to the alert count, tests significance against a coin
flip using the day count as the effective sample, and refuses to recommend a cut until a
setup has enough independent days behind it.

It grades on the underlying's move, not the option's, because option inputs (delta,
theta, ask) were only journalled from 2026-09-03 -- everything earlier can never be
scored as an option. Stock direction is the honest common denominator. Once /took
confirmations exist, real fills replace all of this; see the note at the bottom.

    python scripts/check_setup_performance.py [--horizon 3] [--min-days 20]
"""
import argparse
import asyncio
import os
import sqlite3
import sys

import aiohttp
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

# Bars come from Tradier, so the token has to be loaded before the strategy is built --
# without it every fetch returns 401 and the report claims there is no price history.
try:
    from dotenv import load_dotenv
    for _env in (os.path.join(_ROOT, ".env"), os.path.join(_ROOT, "config", ".env")):
        if os.path.exists(_env):
            load_dotenv(_env)
except ImportError:
    pass

from src.strategy.donchian_daily import DonchianSwingStrategy  # noqa: E402

DB = os.path.join(_ROOT, "data", "rallyhunter.db")


def load_alerts():
    con = sqlite3.connect(os.path.abspath(DB))
    cols = "timestamp,ticker,direction,price,strategy"
    frames = []
    for table in ("trade_log", "trade_log_archive"):
        try:
            frames.append(pd.read_sql(f"select {cols} from {table}", con))
        except Exception:
            pass
    con.close()
    if not frames:
        return pd.DataFrame()
    a = pd.concat(frames)
    a["date"] = pd.to_datetime(a["timestamp"], errors="coerce").dt.normalize()
    return a.dropna(subset=["date", "price"])


async def forward_moves(alerts, horizon):
    """Signed move in the alerted direction, `horizon` sessions after the alert."""
    don = DonchianSwingStrategy()
    closes = {}
    sem = asyncio.Semaphore(6)
    async with aiohttp.ClientSession() as session:
        async def one(ticker):
            async with sem:
                df = await don.fetch_daily_bars(session, ticker)
                if df is not None and not df.empty:
                    closes[ticker] = df["close"]
        await asyncio.gather(*[one(t) for t in alerts["ticker"].unique()])

    rows = []
    for _, r in alerts.iterrows():
        series = closes.get(r["ticker"])
        if series is None:
            continue
        fwd = series[series.index > r["date"]]
        if len(fwd) < horizon:
            continue
        sign = 1 if r["direction"] == "Long" else -1
        rows.append({
            "setup": r["strategy"],
            "day": r["date"].date(),
            "move": (fwd.iloc[horizon - 1] - r["price"]) / r["price"] * 100 * sign,
        })
    return pd.DataFrame(rows)


def report(d, horizon, min_days):
    try:
        from scipy import stats
    except ImportError:
        stats = None

    print("=" * 88)
    print(f"PER-SETUP PERFORMANCE  --  underlying move {horizon} session(s) after the alert")
    print("=" * 88)
    print(f"{'setup':<34}{'alerts':>7}{'days':>6}{'win%':>7}{'mean':>9}{'p':>8}  {'verdict'}")
    print("-" * 88)

    for setup, g in sorted(d.groupby("setup"), key=lambda kv: kv[1]["move"].mean()):
        n, days = len(g), g["day"].nunique()
        wins = int((g["move"] > 0).sum())
        win_pct = 100 * wins / n

        # Significance on the DAY count, not the alert count. Alerts on one day are one
        # observation of one market, so testing on n would manufacture confidence.
        p = float("nan")
        if stats is not None and days >= 5:
            p = stats.binomtest(round(wins * days / n), days, 0.5).pvalue

        if days < min_days:
            verdict = f"insufficient ({days}/{min_days} days)"
        elif p == p and p < 0.05 and g["move"].mean() < 0:
            verdict = "CUT -- add to SETUP_BLOCKLIST"
        elif g["move"].mean() > 0:
            verdict = "keep"
        else:
            verdict = "watch"

        p_txt = f"{p:.3f}" if p == p else "  --"
        print(f"{setup[:32]:<34}{n:>7}{days:>6}{win_pct:>6.1f}%{g['move'].mean():>+8.2f}%{p_txt:>8}  {verdict}")

    print("-" * 88)
    total_days = d["day"].nunique()
    print(f"  {len(d)} alerts across {total_days} distinct days.")
    if total_days < min_days:
        print(f"  Nothing here is conclusive: {total_days} days of alerting is not enough to")
        print(f"  retire a setup. Come back at {min_days}+ days per setup.")
    print()
    print("  Grades the underlying, not the contract. The stock is roughly a coin flip")
    print("  while the option carries ~1.3%/day of theta at ~11x leverage, so a setup that")
    print("  looks flat here still loses money as an option.")
    print("  This measures ALERTS, not trades. No alert has ever been confirmed with")
    print("  /took, so none of it is evidence about money. Confirm real trades and this")
    print("  becomes a report about your account instead of about a simulation.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3,
                    help="sessions after the alert to measure (default 3, the hold ceiling)")
    ap.add_argument("--min-days", type=int, default=20,
                    help="independent days a setup needs before a cut is suggested")
    args = ap.parse_args()

    alerts = load_alerts()
    if alerts.empty:
        print("No alerts journalled yet.")
        return
    d = asyncio.run(forward_moves(alerts, args.horizon))
    if d.empty:
        print("No alerts have enough forward price history yet.")
        return
    report(d, args.horizon, args.min_days)


if __name__ == "__main__":
    main()
