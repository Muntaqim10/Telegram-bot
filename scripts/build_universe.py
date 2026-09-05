"""Rebuild the eligible universe from an independent source.

CANDIDATE_POOL is the alert gate -- signal_pipeline blocks anything outside it -- and a
hand-written 255-ticker list was costing the bot the market. Screened against 4,728
primary-exchange common stocks from Finnhub (a different provider from the one supplying
prices), narrowed to 1,169 liquid names with full-year history:

    top  25 movers of 2026  ->   3 in the pool  (12%)
    top 100                 ->   9              ( 9%)

IRON (+1268%), AEHR (+289%) and AXTI (+268%) were never subscribed, so no gate, model or
strategy downstream could have alerted them.

Three screens, in increasing cost:

  1. Finnhub  -- primary-exchange common stock, plain tickers. Independent of Tradier,
                 so the universe is not defined by the same feed that prices it.
  2. Tradier  -- bulk quotes for price and average volume. One call per 100 names.
  3. Tradier  -- the option chain in the configured DTE window, tested against the bot's
                 OWN liquidity rule. A name whose contracts it would reject anyway is
                 not eligible, it is just noise. On the top 60 movers this passed 57%,
                 against 90% for the current pool -- mostly small biotech failing.

Writes config/universe.txt, which dynamic_scanner loads on import. Tracked in git, unlike
data/, so the pool that produced a given set of alerts is recoverable.

    ./.venv/Scripts/python.exe scripts/build_universe.py [--min-price 3] [--min-volume 500000]
                                                         [--limit 600] [--dry-run]

Note: Tradier's week_52_high/low are NOT reverse-split adjusted -- an early version of
this screen surfaced a $4.70 stock with a $148,800 52-week high. Nothing here uses them.
"""
import argparse
import asyncio
import json
import os
import sys
import urllib.parse
import urllib.request

import aiohttp

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv
    for _env in (os.path.join(_ROOT, ".env"), os.path.join(_ROOT, "config", ".env")):
        if os.path.exists(_env):
            load_dotenv(_env)
except ImportError:
    pass

OUT = os.path.join(_ROOT, "config", "universe.txt")
TRADIER = "https://api.tradier.com/v1"


def finnhub_symbols(token):
    """Primary-exchange US common stock. The independent half of the screen."""
    url = "https://finnhub.io/api/v1/stock/symbol?" + urllib.parse.urlencode(
        {"exchange": "US", "token": token})
    with urllib.request.urlopen(url, timeout=60) as r:
        rows = json.load(r)
    keep = []
    for s in rows:
        if s.get("type") != "Common Stock":
            continue
        if s.get("mic") not in ("XNAS", "XNYS", "ARCX", "BATS"):
            continue
        sym = (s.get("symbol") or "").strip().upper()
        if sym.isalpha() and 1 <= len(sym) <= 5:
            keep.append(sym)
    return sorted(set(keep))


def tradier_liquidity(symbols, headers, min_price, min_volume):
    """Price and average volume, 100 names per call."""
    out = {}
    for i in range(0, len(symbols), 100):
        q = urllib.parse.urlencode({"symbols": ",".join(symbols[i:i + 100]), "greeks": "false"})
        req = urllib.request.Request(f"{TRADIER}/markets/quotes?{q}")
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
        except Exception:
            continue
        quotes = (data.get("quotes") or {}).get("quote") or []
        if isinstance(quotes, dict):
            quotes = [quotes]
        for x in quotes:
            last, vol = x.get("last") or 0, x.get("average_volume") or 0
            if last >= min_price and vol >= min_volume:
                out[x["symbol"]] = {"last": last, "avgvol": vol}
    return out


async def option_tradeable(session, pricer, ticker, headers):
    """Would the bot's own liquidity rule accept any contract in its DTE window?"""
    try:
        expiration = await pricer.get_target_expiration(ticker, session=session)
        if not expiration:
            return False
        url = (f"{TRADIER}/markets/options/chains?symbol={ticker}"
               f"&expiration={expiration}&greeks=true")
        async with session.get(url, headers=headers, timeout=20) as r:
            if r.status != 200:
                return False
            data = await r.json()
    except Exception:
        return False
    contracts = ((data.get("options") or {}) or {}).get("option") or []
    if isinstance(contracts, dict):
        contracts = [contracts]
    for c in contracts:
        bid, ask = c.get("bid") or 0, c.get("ask") or 0
        mid = (bid + ask) / 2
        if mid <= 0:
            continue
        spread = ask - bid
        # The same rule as options_pricer.find_optimal_contract.
        tight = (spread / mid <= 0.18) or (spread <= 0.35 and mid >= 3.0)
        if ((c.get("open_interest") or 0) >= 50 or (c.get("volume") or 0) >= 20) and tight:
            return True
    return False


async def screen_options(candidates, token, limit):
    from src.execution.options_pricer import OptionsPricer
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    pricer = OptionsPricer(token)
    ranked = sorted(candidates.items(), key=lambda kv: -kv[1]["avgvol"])[:limit]
    sem = asyncio.Semaphore(5)
    keep = []
    async with aiohttp.ClientSession() as session:
        async def one(ticker):
            async with sem:
                if await option_tradeable(session, pricer, ticker, headers):
                    keep.append(ticker)
        await asyncio.gather(*[one(t) for t, _ in ranked])
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-price", type=float, default=3.0)
    ap.add_argument("--min-volume", type=float, default=500_000)
    ap.add_argument("--limit", type=int, default=600,
                    help="most liquid N to option-screen; each costs two API calls")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    ftok, ttok = os.getenv("FINNHUB_TOKEN"), os.getenv("TRADIER_ACCESS_TOKEN")
    if not ftok or not ttok:
        print("Need FINNHUB_TOKEN and TRADIER_ACCESS_TOKEN in .env.")
        return 1

    print("1/3  Finnhub: primary-exchange common stock...")
    symbols = finnhub_symbols(ftok)
    print(f"     {len(symbols)}")

    print(f"2/3  Tradier: price >= ${args.min_price:g}, avg volume >= {args.min_volume:,.0f}...")
    headers = {"Authorization": f"Bearer {ttok}", "Accept": "application/json"}
    liquid = tradier_liquidity(symbols, headers, args.min_price, args.min_volume)
    print(f"     {len(liquid)}")

    print(f"3/3  Option chains: the bot's own liquidity rule, top {args.limit} by volume...")
    keep = asyncio.run(screen_options(liquid, ttok, args.limit))
    print(f"     {len(keep)} tradeable")

    from src.data.dynamic_scanner import _DEFAULT_CANDIDATE_POOL
    added = sorted(set(keep) - set(_DEFAULT_CANDIDATE_POOL))
    print()
    print(f"  built-in pool : {len(_DEFAULT_CANDIDATE_POOL)}")
    print(f"  screened      : {len(keep)}")
    print(f"  newly eligible: {len(added)}")
    print(f"  sample        : {', '.join(added[:14])}")

    if len(keep) < len(_DEFAULT_CANDIDATE_POOL):
        print("\n  Screen returned fewer names than the built-in pool; not writing.")
        print("  dynamic_scanner would ignore it anyway. Check API limits and rerun.")
        return 1
    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Generated by scripts/build_universe.py -- the alert gate.\n")
        f.write("# Finnhub primary-exchange common stock, screened for price, volume,\n")
        f.write("# and a contract the bot would actually accept. Regenerate monthly.\n")
        f.write(f"# {len(keep)} tickers; the built-in pool is merged in at load.\n")
        f.write("\n".join(keep) + "\n")
    print(f"\n  wrote {OUT}")
    print("  Restart the engine to pick it up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
