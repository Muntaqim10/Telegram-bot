"""Is there anything for the model to learn?

Before asking whether the model is calibrated, ask whether the features predict the
outcome at all. This measures out-of-sample AUC on time-ordered splits, which is the
question "does a setup's score tell me anything about how it resolves?"

AUC 0.50 means no signal. A model trained on no signal will either predict the base
rate for everything (honest, and useless as a filter) or invent confident-looking
scores by fitting noise (dishonest, and worse than useless).

Run: python scripts/check_feature_signal.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PARQUET = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ml_training_data_v2.parquet"))
FEATURE_JSON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models/feature_columns.json"))

# Known only after the trade closes -- including these would leak the answer.
LEAKY = {"option_final_price", "option_pnl_pct", "max_gain_pct", "days_held", "target"}


def main():
    if not os.path.exists(PARQUET):
        print(f"No training data at {PARQUET}")
        return 1

    df = pd.read_parquet(PARQUET)
    if "date" in df.columns:
        df = df.sort_values("date")
    df = df.reset_index(drop=True)
    y = df["target"].values

    live = ["sma_spread", "sma20_ratio", "rsi_14", "direction_code"]
    if os.path.exists(FEATURE_JSON):
        import json
        try:
            live = json.load(open(FEATURE_JSON))
        except Exception:
            pass

    print(f"rows {len(df)}   base rate {y.mean():.4f}   live features: {live}")

    num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in LEAKY if c in df.columns], errors="ignore")

    print("\nunivariate correlation with outcome (non-leaky only)")
    corr = sorted(((abs(num[c].corr(df["target"])), num[c].corr(df["target"]), c)
                   for c in num.columns if pd.notna(num[c].corr(df["target"]))), reverse=True)
    for _, r, c in corr[:8]:
        print(f"  {c:<22} {r:+.4f}")

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("\nscikit-learn not installed; skipping the AUC test.")
        return 0

    def evaluate(name, X):
        cv = TimeSeriesSplit(n_splits=5)
        best = 0.0
        for label, model in (
            ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
            ("gbdt", GradientBoostingClassifier(random_state=0)),
        ):
            try:
                s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
                best = max(best, s.mean())
                print(f"  {name:<30} {label:<7} AUC {s.mean():.4f} +/- {s.std():.4f}")
            except Exception as e:
                print(f"  {name:<30} {label:<7} failed: {e}")
        return best

    print("\nout-of-sample AUC, time-ordered splits (0.50 = no signal)")
    best = evaluate("the live features", df[live].values)
    best = max(best, evaluate(f"all {num.shape[1]} non-leaky numeric", num.fillna(0).values))

    cats = [c for c in ("asset_tier", "catalyst_type", "direction") if c in df.columns]
    if cats:
        wide = pd.concat([num.fillna(0),
                          pd.get_dummies(df[cats].astype(str), drop_first=True)], axis=1)
        best = max(best, evaluate(f"+ categoricals ({wide.shape[1]} cols)", wide.values))

    print("\n" + "=" * 70)
    if best < 0.55:
        print(f"VERDICT: best out-of-sample AUC {best:.4f}. There is no usable entry signal")
        print("in these features. A model trained on this can only predict the base rate;")
        print("any model that produces a confident spread is fitting noise, and will look")
        print("anti-predictive on live trades. Do not gate alerts on its score.")
    elif best < 0.60:
        print(f"VERDICT: best AUC {best:.4f} -- marginal. Real but small; size the gates")
        print("accordingly and re-check on live outcomes before trusting the tiers.")
    else:
        print(f"VERDICT: best AUC {best:.4f} -- usable signal. Verify it survives on live")
        print("outcomes via scripts/check_live_calibration.py.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
