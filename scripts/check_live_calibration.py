"""Did the bot's predictions come true — measured against real money?

Three sources of truth, in descending order of how much they should be believed:

  1. REAL FILLS      what the trader actually paid and received, from the Robinhood
                     statement and from /closed. This is the only one that is money.
  2. SIMULATED       what the bot's trailing stop would have done on stock ticks. It
                     never bought anything; treat it as a counterfactual, not a result.
  3. BACKTEST        historical simulation. Furthest from reality; not read here.

Anything measured on (2) says whether the signal was directionally right. Only (1) says
whether following it made money.

Run: python scripts/check_live_calibration.py
"""
import os
import sqlite3
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/rallyhunter.db"))

# Predictions either side of a retrain are not comparable.
RETRAIN_AT = os.environ.get("RETRAIN_AT", "2026-08-27 23:27")
# Below this, differences between buckets are noise.
MIN_CONCLUSIVE = 30


def tier(conviction):
    return (conviction or "UNSET").replace("🟢 ", "").replace("🟡 ", "") \
                                  .replace("🔴 ", "").replace("⚪ ", "").strip()


def load(conn):
    """Every alert, keyed by (table, id) because the two tables autoincrement separately."""
    alerts = {}
    for table in ("trade_log", "trade_log_archive"):
        try:
            for r in conn.execute(f"SELECT * FROM {table}"):
                d = dict(r)
                d["_table"] = table
                alerts[(table, d["id"])] = d
        except sqlite3.Error:
            continue
    return alerts


def block(title, rows, value_key, unit="%"):
    if not rows:
        print(f"  {title:<26} —")
        return
    wins = [r for r in rows if r[value_key] > 0]
    vals = [r[value_key] for r in rows]
    print(f"  {title:<26} n={len(rows):>4}  win {len(wins)/len(rows)*100:>5.1f}%  "
          f"mean {statistics.mean(vals)*100:>+7.1f}{unit}  "
          f"median {statistics.median(vals)*100:>+7.1f}{unit}")


def main():
    if not os.path.exists(DB):
        print(f"No database at {DB}")
        return 1
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    cols = {r[1] for r in conn.execute("PRAGMA table_info(real_fills)")} \
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='real_fills'").fetchone() \
        else set()
    if "matched_alert_table" not in cols:
        print("real_fills is missing matched_alert_table — run src.database._init_db_sync() "
              "and re-run scripts/import_robinhood.py")
        return 1

    alerts = load(conn)
    fills = [dict(r) for r in conn.execute(
        "SELECT * FROM real_fills WHERE pnl_pct IS NOT NULL")]
    conn.close()

    took, own = [], []
    for f in fills:
        key = (f.get("matched_alert_table"), f.get("matched_alert_id"))
        a = alerts.get(key)
        if a:
            f["_alert"] = a
            took.append(f)
        else:
            own.append(f)

    print("=" * 78)
    print("1. REAL MONEY — fills the trader actually made")
    print("=" * 78)
    print(f"  decided contracts: {len(fills)}   traced to an alert: {len(took)}   "
          f"self-directed: {len(own)}\n")
    block("Taken from an alert", took, "pnl_pct")
    block("Own picks", own, "pnl_pct")

    if len(took) < MIN_CONCLUSIVE:
        print(f"\n  Only {len(took)} alert-driven fills. Nothing below this line is")
        print(f"  conclusive until that reaches ~{MIN_CONCLUSIVE}. Use /took and /closed,")
        print("  and re-run scripts/import_robinhood.py as the statement grows.")

    if took:
        print("\n  by conviction tier at alert time")
        by_tier = defaultdict(list)
        for f in took:
            by_tier[tier(f["_alert"].get("conviction"))].append(f)
        for t in ("HIGH", "MEDIUM", "LOW", "UNSCORED", "UNSET"):
            if by_tier[t]:
                block(f"  {t}", by_tier[t], "pnl_pct")

        print("\n  how long after the alert the trade was entered")
        by_lag = defaultdict(list)
        for f in took:
            lag = f.get("alert_lag_days")
            by_lag["same day" if lag == 0 else f"{lag}d later" if lag is not None else "?"].append(f)
        for k in sorted(by_lag):
            block(f"  {k}", by_lag[k], "pnl_pct")

    # ---------------------------------------------------------------- simulated
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    sim = []
    for r in conn.execute("SELECT * FROM trade_log_archive WHERE outcome IS NOT NULL"):
        d = dict(r)
        entry, exit_ = d.get("price"), d.get("exit_price")
        if not entry or entry <= 0 or exit_ is None:
            continue
        d["pnl_pct"] = ((exit_ - entry) / entry) if d.get("direction") == "Long" \
            else ((entry - exit_) / entry)
        sim.append(d)
    conn.close()

    print("\n" + "=" * 78)
    print("2. SIMULATED — what the trailing stop would have done. Not money.")
    print("=" * 78)
    print(f"  closed simulations: {len(sim)}\n")
    if sim:
        block("All", sim, "pnl_pct")
        by_tier = defaultdict(list)
        for d in sim:
            by_tier[tier(d.get("conviction"))].append(d)
        print()
        for t in ("HIGH", "MEDIUM", "LOW", "UNSCORED", "UNSET"):
            if by_tier[t]:
                block(f"  {t}", by_tier[t], "pnl_pct")

        before = [d for d in sim if (d.get("timestamp") or "") <= RETRAIN_AT]
        after = [d for d in sim if (d.get("timestamp") or "") > RETRAIN_AT]
        print(f"\n  retrain split at {RETRAIN_AT}: {len(before)} before, {len(after)} after")
        after_preds = [d["xgb_win_prob"] for d in after if d.get("xgb_win_prob") is not None]
        for label, p in (("before", [d["xgb_win_prob"] for d in before
                                     if d.get("xgb_win_prob") is not None]),
                         ("after ", after_preds)):
            if p:
                print(f"    {label}: {len(p):>4} predictions, {len(set(p)):>3} distinct, "
                      f"range {min(p):.4f}-{max(p):.4f}")
        if after_preds and (len(set(after_preds)) <= 5 or max(after_preds) - min(after_preds) < 0.02):
            print("    *** the current model emits a near-constant score on live alerts.")
            print("        Conviction tiers rank nothing. See scripts/check_feature_signal.py.")

    print("\n" + "=" * 78)
    if len(took) >= MIN_CONCLUSIVE and own:
        tw = len([f for f in took if f["pnl_pct"] > 0]) / len(took) * 100
        ow = len([f for f in own if f["pnl_pct"] > 0]) / len(own) * 100
        print(f"VERDICT: alert-driven {tw:.1f}% vs self-directed {ow:.1f}% — {tw-ow:+.1f} points")
    else:
        print("VERDICT: not enough alert-driven fills to judge whether the bot helps.")
        print(f"         {len(took)} traced so far; ~{MIN_CONCLUSIVE} would start to mean something.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
