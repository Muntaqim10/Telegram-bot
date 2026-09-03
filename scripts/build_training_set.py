"""Build a training set from the trader's REAL fills, not from a backtest.

Everything the bot has learned from until now came from simulated backtest outcomes.
This joins each actual Robinhood fill to the market state at the moment it was opened,
so the label is real money rather than a hypothetical trailing stop.

What can be reconstructed after the fact, and what cannot:

  technical analysis   YES -- recomputed from daily bars at the entry date
  XGB sentinel score   YES -- the model takes those same TA features
  contract structure   YES -- DTE, moneyness and premium come from the statement
  option greeks        NO  -- historical chains are not available; delta/theta/IV at
                              entry were never recorded and cannot be recovered
  news sentiment       NO  -- old headlines are not reliably retrievable, and scoring
                              them today would leak hindsight into a past decision

Greeks and sentiment are therefore captured going forward instead (see the alert
capture path), which is why this script reports feature coverage rather than silently
training on whatever happens to be present.

  python scripts/build_training_set.py --bars path/to/bars.json
  python scripts/build_training_set.py --bars bars.json --out data/real_training_set.parquet
"""
import argparse
import datetime
import json
import os
import sqlite3
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/rallyhunter.db"))


def load_bars(path):
    raw = json.load(open(path, encoding="utf-8"))
    out = {}
    for ticker, bars in raw.items():
        clean = []
        for b in bars:
            try:
                o, h, l, c = float(b["open"]), float(b["high"]), float(b["low"]), float(b["close"])
                v = float(b.get("volume") or 0)
            except (TypeError, ValueError, KeyError):
                continue
            if o <= 0 or c <= 0 or h < l:
                continue
            clean.append({"date": b["date"], "o": o, "h": h, "l": l, "c": c, "v": v})
        clean.sort(key=lambda x: x["date"])
        if len(clean) > 60:
            out[ticker] = clean
    return out


def index_on_or_before(bars, date_str):
    lo, hi, best = 0, len(bars) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if bars[mid]["date"] <= date_str:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def features_at(bars, i):
    """Market state on the entry bar. Uses only bars up to and including i."""
    if i is None or i < 60:
        return None
    w = bars[: i + 1]
    close = w[-1]["c"]

    sma20 = statistics.mean(b["c"] for b in w[-20:])
    sma60 = statistics.mean(b["c"] for b in w[-60:])

    gains, losses = [], []
    for k in range(-14, 0):
        d = w[k]["c"] - w[k - 1]["c"]
        (gains if d > 0 else losses).append(abs(d))
    avg_gain = sum(gains) / 14 if gains else 0.0
    avg_loss = sum(losses) / 14 if losses else 0.0
    rsi = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

    trs = []
    for k in range(-14, 0):
        prev = w[k - 1]["c"]
        trs.append(max(w[k]["h"] - w[k]["l"], abs(w[k]["h"] - prev), abs(w[k]["l"] - prev)))
    atr = sum(trs) / len(trs)

    vols = [b["v"] for b in w[-21:-1] if b["v"] > 0]
    vmean = statistics.mean(vols) if vols else 0.0
    vstd = statistics.pstdev(vols) if len(vols) > 1 else 0.0
    z_vol = (w[-1]["v"] - vmean) / vstd if vstd > 0 else 0.0

    run = 0
    for b in reversed(w):
        green = b["c"] > b["o"]
        if run == 0:
            run = 1 if green else -1
        elif (run > 0) == green:
            run += 1 if green else -1
        else:
            break

    return {
        "close": round(close, 4),
        "sma20_ratio": round(close / sma20, 4) if sma20 > 0 else 1.0,
        "sma_spread": round((sma20 - sma60) / sma60, 4) if sma60 > 0 else 0.0,
        "rsi_14": round(rsi, 2),
        "atr_pct": round(atr / close, 4) if close > 0 else 0.0,
        "z_vol": round(z_vol, 3),
        "candle_run": run,
        "ret_5d": round((close - w[-6]["c"]) / w[-6]["c"], 4) if len(w) > 6 else 0.0,
        "ret_20d": round((close - w[-21]["c"]) / w[-21]["c"], 4) if len(w) > 21 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bars", required=True, help="JSON of {ticker: [daily bars]}")
    ap.add_argument("--out", default=None, help="write the table to this parquet/csv path")
    args = ap.parse_args()

    if not os.path.exists(args.bars):
        print(f"No bars file at {args.bars}")
        return 1
    bars = load_bars(args.bars)
    print(f"bars loaded for {len(bars)} tickers")

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    fills = [dict(r) for r in conn.execute(
        "SELECT * FROM real_fills WHERE pnl_pct IS NOT NULL ORDER BY open_date")]
    conn.close()
    print(f"real fills with a known result: {len(fills)}")

    try:
        from src.ai.xgb_micro_v2 import XGBMicroSentinelV2
        model = XGBMicroSentinelV2()
    except Exception as e:
        print(f"(XGB model unavailable: {e})")
        model = None

    rows, skipped = [], {"no bars": 0, "no history": 0}
    for f in fills:
        series = bars.get(f["ticker"])
        if not series:
            skipped["no bars"] += 1
            continue
        feats = features_at(series, index_on_or_before(series, f["open_date"]))
        if feats is None:
            skipped["no history"] += 1
            continue

        direction_code = 1 if (f["option_type"] or "").upper() == "CALL" else 0
        row = dict(feats)
        row.update({
            "ticker": f["ticker"],
            "open_date": f["open_date"],
            "option_type": f["option_type"],
            "direction_code": direction_code,
            "strike": f["strike"],
            "premium": f["entry_price"],
            "cost": f["cost"],
            "quantity": f["quantity"],
            "days_held": f["days_held"],
            "exit_kind": f["exit_kind"],
            # contract structure, which the statement does record
            "dte_at_entry": None,
            "moneyness": round(f["strike"] / feats["close"], 4) if feats["close"] else None,
            "premium_pct_of_spot": round((f["entry_price"] or 0) / feats["close"], 4) if feats["close"] else None,
            # greeks and sentiment were never captured for these trades
            "delta": None, "theta": None, "iv": None, "sentiment": None,
            # labels, from real money
            "pnl_pct": f["pnl_pct"],
            "target": 1 if f["pnl_pct"] > 0 else 0,
            "target_50": 1 if f["pnl_pct"] >= 0.50 else 0,
        })
        try:
            row["dte_at_entry"] = (datetime.date.fromisoformat(f["expiration"])
                                   - datetime.date.fromisoformat(f["open_date"])).days
        except (TypeError, ValueError):
            pass
        if model and model.is_active:
            r = model.validate_setup({
                "sma_spread": feats["sma_spread"], "sma20_ratio": feats["sma20_ratio"],
                "rsi_14": feats["rsi_14"], "direction_code": direction_code})
            row["xgb_win_prob"] = round(r["win_prob"], 6)
            row["xgb_verdict"] = r["verdict"]
        rows.append(row)

    print(f"built {len(rows)} training rows  (skipped: {skipped})\n")
    if not rows:
        print("Nothing to work with.")
        return 1

    print("FEATURE COVERAGE")
    for col in ("rsi_14", "sma20_ratio", "sma_spread", "atr_pct", "z_vol", "candle_run",
                "dte_at_entry", "moneyness", "premium_pct_of_spot", "xgb_win_prob",
                "delta", "theta", "iv", "sentiment"):
        present = sum(1 for r in rows if r.get(col) is not None)
        state = "reconstructed" if present else "NOT RECOVERABLE - capture going forward"
        print(f"  {col:<22} {present:>4}/{len(rows)}  {state}")
    print()

    wins = sum(r["target"] for r in rows)
    print(f"LABELS: {wins}/{len(rows)} winners ({wins/len(rows)*100:.1f}%), "
          f"{sum(r['target_50'] for r in rows)} reached +50%")
    print(f"        mean return {statistics.mean(r['pnl_pct'] for r in rows)*100:+.1f}%")

    if args.out:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            if args.out.endswith(".parquet"):
                df.to_parquet(args.out, index=False)
            else:
                df.to_csv(args.out, index=False)
            print(f"\nwrote {len(df)} rows x {len(df.columns)} cols -> {args.out}")
        except Exception as e:
            print(f"\nCould not write {args.out}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
