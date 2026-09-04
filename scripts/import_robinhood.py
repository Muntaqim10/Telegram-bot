"""Import a Robinhood statement and reconcile it against what the bot alerted.

The bot cannot see the brokerage account, so everything it records is either a
suggestion (an alert) or a simulation (a trailing stop run on stock ticks). This
script imports what actually happened and matches it back to the alerts, which
answers the only question that matters: are the bot's alerts worth taking?

  python scripts/import_robinhood.py path/to/statement.csv
  python scripts/import_robinhood.py path/to/statement.csv --dry-run
  python scripts/import_robinhood.py path/to/statement.csv --match-window 5

Re-running is safe: fills are keyed on (contract, open_date) and replaced in place.
"""
import argparse
import csv
import datetime
import os
import re
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.database as db

# "PLTR 9/18/2026 Call $150.00", optionally prefixed ("Option Expiration for ...").
CONTRACT = re.compile(r"([A-Z]+)\s+(\d{1,2}/\d{1,2}/\d{4})\s+(Call|Put)\s+\$([\d,.]+)")

OPEN_CODES = {"BTO"}
SELL_CODES = {"STC"}
EXPIRE_CODES = {"OEXP"}
EXERCISE_CODES = {"OEXCS", "OASGN"}


def money(raw):
    """Robinhood writes debits as ($1,234.56)."""
    s = (raw or "").strip().replace("$", "").replace(",", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if not s:
        return 0.0
    try:
        v = float(s)
    except ValueError:
        return 0.0
    return -v if neg else v


def parse_statement(path):
    """Rows, oldest first. Quantity carries an 'S' suffix on short legs."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw_date = (r.get("Activity Date") or "").strip()
            if not raw_date:
                continue
            try:
                d = datetime.datetime.strptime(raw_date, "%m/%d/%Y").date()
            except ValueError:
                continue
            qty = (r.get("Quantity") or "").strip()
            rows.append({
                "date": d,
                "desc": (r.get("Description") or "").strip(),
                "code": (r.get("Trans Code") or "").strip(),
                "short_leg": qty.endswith("S"),
                "qty": float(qty.rstrip("S")) if re.match(r"^[\d.]+S?$", qty) else 0.0,
                "price": money(r.get("Price")),
                "amount": money(r.get("Amount")),
            })
    rows.sort(key=lambda x: x["date"])
    return rows


def contract_key(desc):
    m = CONTRACT.search(desc)
    if not m:
        return None
    return (m.group(1), m.group(2), m.group(3), m.group(4).replace(",", ""))


def build_fills(rows):
    """Collapse the transaction log into one record per long option contract."""
    legs = defaultdict(lambda: {"open": [], "sell": [], "expire": [], "exercise": []})
    for r in rows:
        key = contract_key(r["desc"])
        if not key:
            continue
        if r["code"] in OPEN_CODES:
            legs[key]["open"].append(r)
        elif r["code"] in SELL_CODES:
            legs[key]["sell"].append(r)
        elif r["code"] in EXPIRE_CODES and not r["short_leg"]:
            legs[key]["expire"].append(r)
        elif r["code"] in EXERCISE_CODES and not r["short_leg"]:
            legs[key]["exercise"].append(r)

    fills = []
    for key, v in legs.items():
        if not v["open"]:
            continue  # short premium or a leg we never bought; out of scope here
        ticker, exp_raw, opt_type, strike = key
        cost = -sum(x["amount"] for x in v["open"])
        if cost <= 0:
            continue
        qty = sum(x["qty"] for x in v["open"]) or 1.0
        open_date = min(x["date"] for x in v["open"])
        entry_price = (v["open"][0]["price"] or (cost / qty / 100.0))

        proceeds, close_date, exit_price, kind = 0.0, None, None, "OPEN"
        if v["sell"]:
            proceeds = sum(x["amount"] for x in v["sell"])
            close_date = max(x["date"] for x in v["sell"])
            exit_price = v["sell"][0]["price"] or (proceeds / qty / 100.0)
            kind = "SOLD"
        elif v["exercise"]:
            # Value moved into shares or cash-settled; the statement alone cannot price it.
            close_date = max(x["date"] for x in v["exercise"])
            kind = "EXERCISED"
        elif v["expire"]:
            close_date = max(x["date"] for x in v["expire"])
            exit_price, proceeds, kind = 0.0, 0.0, "EXPIRED"

        pnl = pnl_pct = None
        if kind in ("SOLD", "EXPIRED"):
            pnl = proceeds - cost
            pnl_pct = pnl / cost if cost else None

        try:
            expiration = datetime.datetime.strptime(exp_raw, "%m/%d/%Y").date().isoformat()
        except ValueError:
            expiration = exp_raw

        fills.append({
            "contract": f"{ticker} {expiration} {opt_type} {strike}",
            "ticker": ticker,
            "option_type": opt_type.upper(),
            "strike": float(strike),
            "expiration": expiration,
            "open_date": open_date.isoformat(),
            "close_date": close_date.isoformat() if close_date else None,
            "quantity": qty,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4) if exit_price is not None else None,
            "cost": round(cost, 2),
            "proceeds": round(proceeds, 2),
            "pnl": round(pnl, 2) if pnl is not None else None,
            "pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
            "days_held": (close_date - open_date).days if close_date else None,
            "exit_kind": kind,
            "matched_alert_id": None,
            "matched_alert_table": None,
            "alert_lag_days": None,
        })
    fills.sort(key=lambda f: f["open_date"])
    return fills


def load_alerts():
    """Every alert the bot has ever dispatched, open or archived."""
    if not os.path.exists(db.DB_PATH):
        return []
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    out = []
    for table in ("trade_log", "trade_log_archive"):
        try:
            for r in conn.execute(f"SELECT * FROM {table}"):
                d = dict(r)
                d["_table"] = table
                try:
                    d["_date"] = datetime.datetime.strptime(
                        d["timestamp"][:10], "%Y-%m-%d").date()
                except (ValueError, TypeError, KeyError):
                    d["_date"] = None
                out.append(d)
        except sqlite3.Error:
            continue
    conn.close()
    return [a for a in out if a.get("_date")]


def implied_option_type(alert):
    """The contract an alert argues for. Long -> calls, Short -> puts."""
    direction = (alert.get("direction") or "").strip().lower()
    if direction.startswith("long"):
        return "CALL"
    if direction.startswith("short"):
        return "PUT"
    return None


def match(fills, alerts, window_days):
    """Attach each fill to a preceding alert on the same ticker AND the same side.

    Ticker and date alone are not evidence. On the first statement this matched 8
    contracts, of which 5 were the opposite side from the alert -- the bot said puts on
    PLTR, AMZN and AAPL and the account bought calls -- and one GLD alert was matched to
    both a call and a put on the same strike the same day, which no single alert can have
    caused. All 8 were same-day round trips, while the bot proposes 14-21 DTE contracts
    on a multi-day thesis.

    Requiring the side to agree is the cheapest available filter. It is still not proof
    the alert caused the trade: these are liquid names the account trades anyway, so a
    coincidence on ticker, date AND direction remains ordinary. Only /took establishes
    that, which is why the report below counts confirmations separately and never calls
    a statement match alert-driven.
    """
    by_ticker = defaultdict(list)
    for a in alerts:
        by_ticker[(a.get("ticker") or "").upper()].append(a)

    matched = 0
    for f in fills:
        opened = datetime.date.fromisoformat(f["open_date"])
        best, best_lag = None, None
        for a in by_ticker.get(f["ticker"], []):
            side = implied_option_type(a)
            if side is not None and f.get("option_type") and side != f["option_type"]:
                continue
            lag = (opened - a["_date"]).days
            if 0 <= lag <= window_days and (best_lag is None or lag < best_lag):
                best, best_lag = a, lag
        if best is not None:
            # The id alone is ambiguous: both alert tables autoincrement separately.
            f["matched_alert_id"] = best.get("id")
            f["matched_alert_table"] = best.get("_table")
            f["alert_lag_days"] = best_lag
            matched += 1
    return matched


def pct(n, d):
    return f"{n / d * 100:.1f}%" if d else "n/a"


def report(fills, alerts, window_days):
    decided = [f for f in fills if f["pnl_pct"] is not None]
    took = [f for f in decided if f["matched_alert_id"] is not None]

    # Only compare like with like. The statement can span years; the alert log starts
    # when the bot did. Judging alerts against trades from a different market regime
    # (and a different account size) would be meaningless.
    lo = min(a["_date"] for a in alerts) if alerts else None
    hi = max(a["_date"] for a in alerts) if alerts else None
    in_window = [f for f in decided
                 if lo and lo <= datetime.date.fromisoformat(f["open_date"]) <= hi]
    own_same_period = [f for f in in_window if f["matched_alert_id"] is None]
    own_all_time = [f for f in decided if f["matched_alert_id"] is None]

    print("=" * 74)
    print("STATEMENT IMPORT")
    print("=" * 74)
    print(f"  contracts parsed:      {len(fills)}")
    print(f"  with a known result:   {len(decided)}   "
          f"(excludes {len(fills) - len(decided)} exercised/open)")
    print(f"  bot alerts on record:  {len(alerts)}" + (f"  ({lo} .. {hi})" if lo else ""))
    print(f"  contracts opened in that window: {len(in_window)}")
    print(f"  possible alert matches:{len(took)}  (same ticker, same side, within "
          f"{window_days} day(s))")
    print("     ^ a coincidence filter, NOT evidence the alert was traded. These are")
    print("       liquid names the account trades anyway. Only /took confirms a trade,")
    print("       and no position has ever been confirmed, so the bot's real track")
    print("       record is 0 trades -- treat every figure below as the account's own.")

    def block(label, group):
        if not group:
            print(f"\n  {label}: none")
            return
        wins = [f for f in group if f["pnl_pct"] > 0]
        total = sum(f["pnl"] for f in group)
        avg = sum(f["pnl_pct"] for f in group) / len(group)
        holds = [f["days_held"] for f in group if f["days_held"] is not None]
        print(f"\n  {label}: {len(group)} contracts")
        print(f"    win rate     {pct(len(wins), len(group))}")
        print(f"    net P&L      ${total:,.2f}")
        print(f"    avg return   {avg * 100:+.1f}%")
        if holds:
            holds.sort()
            print(f"    median hold  {holds[len(holds) // 2]} day(s)")

    print("\n  --- same period, like for like ---")
    block("TAKEN FROM A BOT ALERT", took)
    block("YOUR OWN PICKS, same window", own_same_period)
    print("\n  --- context ---")
    block("YOUR OWN PICKS, all time", own_all_time)

    print("\n" + "=" * 74)
    if took and own_same_period:
        tw = len([f for f in took if f["pnl_pct"] > 0]) / len(took)
        ow = len([f for f in own_same_period if f["pnl_pct"] > 0]) / len(own_same_period)
        gap = (tw - ow) * 100
        smaller = min(len(took), len(own_same_period))
        print(f"VERDICT: alert-driven {tw*100:.1f}% (n={len(took)}) vs self-directed "
              f"{ow*100:.1f}% (n={len(own_same_period)}) -- {gap:+.1f} points")
        if smaller < 20:
            print(f"         Too few trades on one side (n={smaller}) to conclude anything.")
            print("         Keep importing as the sample grows; this is the number to watch.")
        elif gap > 5:
            print("         The alerts are pulling their weight.")
        elif gap < -5:
            print("         Your own picks are doing better than the bot's.")
        else:
            print("         No meaningful difference between following and ignoring the bot.")
    else:
        print("VERDICT: not enough overlap between the statement and the alert history.")
        print("         The alert log only covers the period since the bot started running.")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("statement", help="Robinhood activity CSV export")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument("--match-window", type=int, default=3,
                    help="days after an alert in which an entry counts as taking it (default 3)")
    args = ap.parse_args()

    if not os.path.exists(args.statement):
        print(f"No such file: {args.statement}")
        return 1

    rows = parse_statement(args.statement)
    if not rows:
        print("No dated rows found -- is this a Robinhood activity export?")
        return 1
    print(f"parsed {len(rows)} statement rows "
          f"({rows[0]['date']} .. {rows[-1]['date']})")

    fills = build_fills(rows)
    alerts = load_alerts()
    match(fills, alerts, args.match_window)
    report(fills, alerts, args.match_window)

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    db._init_db_sync()
    db._upsert_real_fills_sync(fills)
    print(f"\nwrote {len(fills)} fill(s) to real_fills in {db.DB_PATH}")
    print("run scripts/check_live_calibration.py to compare predictions against them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
