"""What the alerts would have made, priced at the spread you would actually have paid.

This is the only report in the repo about the alerter that is not a simulation of a
simulation. Every other one is:

  trade_log / trade_outcomes.csv   the trailing stop's counterfactual on positions
                                   nobody took -- 249 "closures", zero confirmed
  ml_training_data_v2.parquet      Black-Scholes on realized vol, over four setups
                                   that never appear in the live alert stream
  real_fills                       the account's own trading, not the bot's

Paper trades are entered at the ask the alert was quoted and closed at the real bid, so
the round trip pays the spread twice and unquotable contracts settle at zero. That makes
the number pessimistic where it is uncertain, which is the correct direction for a
system deciding whether to risk money.

    python scripts/check_paper_book.py [--min-days 20]
"""
import argparse
import os
import sqlite3
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

DB = os.path.join(_ROOT, "data", "rallyhunter.db")


def fetch():
    if not os.path.exists(DB):
        return [], []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    try:
        closed = [dict(r) for r in conn.execute(
            "SELECT * FROM paper_trades WHERE closed_at IS NOT NULL")]
        live = [dict(r) for r in conn.execute(
            "SELECT * FROM paper_trades WHERE closed_at IS NULL")]
    except sqlite3.OperationalError:
        return [], []
    finally:
        conn.close()
    return closed, live


def summarise(label, group):
    n = len(group)
    if not n:
        return
    wins = sum(1 for t in group if (t["pnl_pct"] or 0) > 0)
    pnls = sorted(t["pnl_pct"] or 0.0 for t in group)
    mean = sum(pnls) / n
    median = pnls[n // 2] if n % 2 else (pnls[n // 2 - 1] + pnls[n // 2]) / 2
    days = len({(t["opened_at"] or "")[:10] for t in group})
    print(f"  {label[:32]:34}{n:>5}{days:>6}{100*wins/n:>7.1f}%{mean:>9.1f}%{median:>9.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=int, default=20,
                    help="independent days before the numbers mean anything")
    args = ap.parse_args()

    closed, live = fetch()
    if not closed and not live:
        print("No paper trades yet. They open automatically as alerts dispatch --")
        print("start the engine and check back after a session.")
        return

    print("=" * 76)
    print("PAPER BOOK  --  entered at the ask, closed at the bid, spread paid both ways")
    print("=" * 76)
    print(f"  open: {len(live)}    closed: {len(closed)}")
    if not closed:
        print("\n  Nothing has closed yet, so there is nothing to grade.")
        return

    days = len({(t["opened_at"] or "")[:10] for t in closed})
    print()
    print(f"  {'':34}{'n':>5}{'days':>6}{'win%':>8}{'mean':>9}{'median':>9}")
    print("  " + "-" * 72)
    summarise("ALL", closed)
    print("  " + "-" * 72)
    for setup in sorted({t["setup"] or "(none)" for t in closed}):
        summarise(setup, [t for t in closed if (t["setup"] or "(none)") == setup])
    print("  " + "-" * 72)
    for d in sorted({t["direction"] or "?" for t in closed}):
        summarise(d, [t for t in closed if (t["direction"] or "?") == d])

    zeros = sum(1 for t in closed if (t["exit_bid"] or 0) == 0)
    print()
    print(f"  settled at zero (no bid at close): {zeros} of {len(closed)}")
    print()
    if days < args.min_days:
        print(f"  {days} of {args.min_days} independent days. Alerts fired on the same day")
        print("  move together, so the day count is the honest sample size -- not n.")
        print("  Nothing here is conclusive yet.")
    else:
        print(f"  {days} independent days. Setups with a negative median here are")
        print("  candidates for SETUP_BLOCKLIST; check scripts/check_setup_performance.py")
        print("  for the significance test before cutting.")


if __name__ == "__main__":
    main()
