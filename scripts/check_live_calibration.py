"""Does the model's confidence mean anything on live trades?

Joins each closed trade in trade_log_archive back to the prediction that opened it
(xgb_win_prob, conviction, sentinel_verdict) and reports realized outcomes against it.
This is the loop that was never closed: predictions were logged, outcomes were logged,
and nothing ever compared the two.

Run: python scripts/check_live_calibration.py
"""
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/rallyhunter.db"))
CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trade_outcomes.csv"))

# Retraining the model resets what a prediction means, so trades either side of one are
# not comparable. Set this to the timestamp of the most recent retrain to split them.
RETRAIN_AT = os.environ.get("RETRAIN_AT", "2026-08-27 23:27")


def pnl_pct(direction, entry, exit_):
    if not entry or entry <= 0 or exit_ is None:
        return None
    return (exit_ - entry) / entry if direction == "Long" else (entry - exit_) / entry


def summarize(label, rows):
    n = len(rows)
    if not n:
        return f"  {label:<24} {'--':>5}"
    wins = [r for r in rows if r["pnl"] is not None and r["pnl"] > 0]
    tps = [r for r in rows if r["outcome"] == "TP_HIT"]
    pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
    avg = sum(pnls) / len(pnls) if pnls else 0.0
    pred = sum(r["pred"] for r in rows) / n
    return (f"  {label:<24} {n:>5}  {pred*100:>9.1f}%  {len(wins)/n*100:>9.1f}%  "
            f"{len(tps)/n*100:>7.1f}%  {avg*100:>+8.2f}%")


def main():
    if not os.path.exists(DB):
        print(f"No database at {DB}")
        return 1
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    cols = {r[1] for r in conn.execute("PRAGMA table_info(trade_log_archive)")}
    if "outcome" not in cols:
        print("trade_log_archive has no outcome column -- run src.database._init_db_sync() first.")
        return 1

    rows = []
    for r in conn.execute(
        "SELECT timestamp, ticker, direction, price, exit_price, outcome, xgb_win_prob, "
        "conviction, sentinel_verdict, strategy FROM trade_log_archive WHERE outcome IS NOT NULL"
    ):
        rows.append({
            "timestamp": r["timestamp"],
            "ticker": r["ticker"],
            "direction": r["direction"],
            "outcome": r["outcome"],
            "pred": r["xgb_win_prob"] if r["xgb_win_prob"] is not None else 0.5,
            "conviction": (r["conviction"] or "UNSET").replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", ""),
            "verdict": r["sentinel_verdict"] or "NOT_CHECKED",
            "strategy": r["strategy"] or "?",
            "pnl": pnl_pct(r["direction"], r["price"], r["exit_price"]),
        })

    if not rows:
        print("No closed trades with outcomes yet.")
        return 0

    before = [r for r in rows if (r["timestamp"] or "") <= RETRAIN_AT]
    after = [r for r in rows if (r["timestamp"] or "") > RETRAIN_AT]

    print("=" * 78)
    print(f"LIVE CALIBRATION -- {len(rows)} closed trades with a recorded prediction")
    print(f"  retrain split at {RETRAIN_AT}: {len(before)} before, {len(after)} after")
    if after and len(after) < 50:
        print(f"  WARNING: only {len(after)} trades since the last retrain. Everything below is")
        print("           dominated by predictions the current model did not make.")
    print("=" * 78)
    header = f"  {'bucket':<24} {'n':>5}  {'predicted':>10}  {'realized':>10}  {'TP hit':>8}  {'avg P&L':>9}"
    print(header)
    print("  " + "-" * 74)

    print(summarize("ALL", rows))
    print()

    print("  by conviction tier")
    for tier in ("HIGH", "MEDIUM", "LOW", "UNSET"):
        sub = [r for r in rows if r["conviction"] == tier]
        if sub:
            print(summarize(tier, sub))
    print()

    print("  by model confidence")
    buckets = defaultdict(list)
    for r in rows:
        p = r["pred"]
        if p < 0.36: key = "< 0.360 (below gate)"
        elif p < 0.365: key = "0.360 - 0.365"
        elif p < 0.375: key = "0.365 - 0.375"
        else: key = ">= 0.375 (HIGH gate)"
        buckets[key].append(r)
    for key in ("< 0.360 (below gate)", "0.360 - 0.365", "0.365 - 0.375", ">= 0.375 (HIGH gate)"):
        if buckets[key]:
            print(summarize(key, buckets[key]))
    print()

    print("  by direction")
    for d in ("Long", "Short"):
        sub = [r for r in rows if r["direction"] == d]
        if sub:
            print(summarize(d, sub))
    print()

    print("  by exit reason")
    for o in ("TP_HIT", "STOPPED_OUT", "INVALIDATED"):
        sub = [r for r in rows if r["outcome"] == o]
        if sub:
            pnls = [x["pnl"] for x in sub if x["pnl"] is not None]
            avg = sum(pnls) / len(pnls) * 100 if pnls else 0.0
            print(f"    {o:<14} {len(sub):>4}  ({len(sub)/len(rows)*100:>4.1f}%)  avg P&L {avg:>+7.2f}%")

    # Before anything else is believable, the model has to actually vary its output.
    print()
    print("  model discrimination (does win_prob depend on the setup at all?)")
    collapsed = False
    for label, sub in (("before retrain", before), ("after retrain ", after)):
        if not sub:
            continue
        preds = [r["pred"] for r in sub]
        distinct = len(set(preds))
        mean = sum(preds) / len(preds)
        stdev = (sum((p - mean) ** 2 for p in preds) / len(preds)) ** 0.5
        flag = ""
        if distinct <= 5 or (max(preds) - min(preds)) < 0.02:
            flag = "   <-- COLLAPSED"
            if sub is after:
                collapsed = True
        print(f"    {label}: n={len(preds):>4}  {distinct:>3} distinct  "
              f"range {min(preds):.4f}-{max(preds):.4f}  stdev {stdev:.6f}{flag}")
    if collapsed:
        print()
        print("    *** The CURRENT model emits a near-constant score. Conviction tiers and the")
        print("        win_prob gates carry no information: every setup scores the same, so")
        print("        nothing is filtered and nothing is ranked. Fix this before reading")
        print("        anything into the tier table above.")

    # spread between the best and worst tier is the only number that matters
    tiers = {t: [r for r in rows if r["conviction"] == t] for t in ("HIGH", "MEDIUM", "LOW")}
    rates = {t: (len([r for r in v if r["pnl"] and r["pnl"] > 0]) / len(v)) for t, v in tiers.items() if v}
    print()
    print("=" * 78)
    if len(rates) >= 2:
        best, worst = max(rates, key=rates.get), min(rates, key=rates.get)
        spread = (rates[best] - rates[worst]) * 100
        print(f"VERDICT: {best} wins {rates[best]*100:.1f}%, {worst} wins {rates[worst]*100:.1f}% "
              f"-- spread {spread:+.1f} points")
        if spread < 5:
            print("The conviction tiers are not separating outcomes. The gates are tuned to noise.")
        elif rates.get("HIGH", 0) < rates.get("LOW", 1):
            print("INVERTED: LOW conviction is outperforming HIGH. The model is anti-predictive.")
        else:
            print("Tiers are ordered correctly. Check the sample size before trusting the size of the gap.")
    else:
        print("Only one conviction tier present -- not enough variety to calibrate yet.")
    print("=" * 78)

    if os.path.exists(CSV):
        try:
            import pandas as pd
            df = pd.read_csv(CSV)
            est = df["estimated_option_pnl_pct"].dropna()
            print(f"\noutcome log: {len(df)} rows, {len(est)} with an option P&L estimate")
            print(f"  estimated option P&L: mean {est.mean()*100:+.1f}%, median {est.median()*100:+.1f}%")
            print("  NOTE: delta+theta approximation, never validated against a real fill.")
        except Exception as e:
            print(f"\nCould not read {CSV}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
