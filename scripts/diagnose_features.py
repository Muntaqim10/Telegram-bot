import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Resolve root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

def main():
    parquet_path = os.path.join(ROOT_DIR, "data", "ml_training_data_v2.parquet")
    print("=" * 70)
    print("  DIAGNOSTIC FEATURE & TRAINING DATA ANALYSIS")
    print("=" * 70)
    print(f"Loading dataset: {parquet_path}\n")

    if not os.path.exists(parquet_path):
        print(f"❌ Error: File not found at {parquet_path}")
        return

    raw_df = pd.read_parquet(parquet_path)
    print(f"Total Raw Rows in Parquet: {len(raw_df)}")
    print(f"Columns present in file: {list(raw_df.columns)}\n")

    requested_features = [
        "entry_time_minute", "relative_volume", "vwap_ratio", 
        "ema9_ratio", "ema_trend", "ema_trend_5m", 
        "spy_correlation", "hod_ratio", "lod_ratio"
    ]

    # =========================================================================
    # STEP 1 -- Feature Variance Check
    # =========================================================================
    print("=" * 70)
    print("STEP 1 -- FEATURE VARIANCE CHECK")
    print("=" * 70)
    print(f"{'Feature':<20} | {'Status':<18} | {'Unique':<6} | {'Min':<8} | {'Max':<8} | {'Mean':<8} | {'Std':<8} | {'Top Val (Freq)':<18}")
    print("-" * 110)

    for feat in requested_features:
        if feat not in raw_df.columns:
            print(f"{feat:<20} | {'MISSING IN PARQUET':<18} | {'0':<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A (100.0% missing)':<18}")
            continue

        series = raw_df[feat].dropna()
        n_rows = len(series)
        if n_rows == 0:
            print(f"{feat:<20} | {'ALL NULL':<18} | {'0':<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A (100.0% null)':<18}")
            continue

        n_unique = int(series.nunique())
        f_min = float(series.min())
        f_max = float(series.max())
        f_mean = float(series.mean())
        f_std = float(series.std()) if n_rows > 1 else 0.0

        val_counts = series.value_counts(dropna=False)
        top_val = val_counts.index[0]
        top_count = val_counts.iloc[0]
        top_pct = (top_count / len(raw_df)) * 100.0

        # Classification rule:
        # LIKELY PLACEHOLDER if top value accounts for > 80% of rows or std is near zero (< 1e-4) or n_unique <= 1
        if top_pct > 80.0 or (pd.notna(f_std) and f_std < 1e-4) or n_unique <= 1:
            label = "LIKELY PLACEHOLDER"
        else:
            label = "REAL SIGNAL"

        top_val_str = f"{top_val:.3f}" if isinstance(top_val, (int, float, np.number)) else str(top_val)
        top_str = f"{top_val_str} ({top_pct:.1f}%)"

        print(f"{feat:<20} | {label:<18} | {n_unique:<6} | {f_min:<8.3f} | {f_max:<8.3f} | {f_mean:<8.3f} | {f_std:<8.3f} | {top_str:<18}")

    print("-" * 110)

    # =========================================================================
    # STEP 2 -- Target Balance Check
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2 -- TARGET BALANCE CHECK")
    print("=" * 70)
    if "target" not in raw_df.columns:
        print("❌ 'target' column missing in dataset.")
    else:
        target_series = raw_df["target"].dropna()
        n_pos = int((target_series == 1).sum())
        n_neg = int((target_series == 0).sum())
        total_target = len(target_series)
        pos_pct = (n_pos / total_target * 100.0) if total_target > 0 else 0.0
        neg_pct = (n_neg / total_target * 100.0) if total_target > 0 else 0.0
        calc_scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 0.0

        print(f"Total Target Rows:             {total_target}")
        print(f"Target == 1 (Win):             {n_pos} ({pos_pct:.2f}%)")
        print(f"Target == 0 (Loss):            {n_neg} ({neg_pct:.2f}%)")
        print(f"Calculated scale_pos_weight:   {calc_scale_pos_weight:.8f}  (count(0) / count(1))")
        print(f"Candidate Model Parameter:     2.16233778")
        diff = abs(calc_scale_pos_weight - 2.16233778)
        if diff < 0.05:
            print(f"--> Sanity Check: MATCHES candidate model scale_pos_weight (diff = {diff:.5f})")
        else:
            print(f"--> Sanity Check: MISMATCH with candidate model (diff = {diff:.5f})")

    # =========================================================================
    # STEP 3 -- Per-Feature Correlation with Target
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3 -- PER-FEATURE CORRELATION WITH TARGET")
    print("=" * 70)
    correlations = []
    if "target" in raw_df.columns:
        for feat in requested_features:
            if feat in raw_df.columns:
                valid = raw_df[[feat, "target"]].dropna()
                if len(valid) > 1 and valid[feat].std() > 1e-6:
                    corr = float(valid[feat].corr(valid["target"]))
                    correlations.append((feat, corr))
                else:
                    correlations.append((feat, 0.0))
            else:
                correlations.append((feat, float("nan")))

        # Sort by absolute correlation descending (treating NaN as -1 for sort)
        correlations.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else -1, reverse=True)

        print(f"{'Rank':<5} | {'Feature':<22} | {'Pearson / Point-Biserial Corr':<30} | {'Direction':<12}")
        print("-" * 75)
        for idx, (feat, corr) in enumerate(correlations, 1):
            if np.isnan(corr):
                print(f"{idx:<5} | {feat:<22} | {'N/A (Missing from Parquet)':<30} | {'None':<12}")
            else:
                dir_str = "Positive" if corr > 0 else ("Negative" if corr < 0 else "Zero")
                print(f"{idx:<5} | {feat:<22} | {corr:<30.5f} | {dir_str:<12}")
        print("-" * 75)

    # =========================================================================
    # STEP 4 -- Sample Size Sanity Check
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4 -- SAMPLE SIZE SANITY CHECK")
    print("=" * 70)
    print(f"Total Rows in Parquet: {len(raw_df)}")

    # Check exact filtering logic from src/ai/xgb_micro_v2.py train() method:
    # 1. for col in features: if col not in df.columns: df[col] = 0.0
    # 2. df = df.dropna(subset=features + ["target"])
    print("\nExact xgb_micro_v2.py train() filtering simulation:")
    df_sim = raw_df.copy()
    missing_cols = []
    for col in requested_features:
        if col not in df_sim.columns:
            missing_cols.append(col)
            df_sim[col] = 0.0

    target_col = ["target"] if "target" in df_sim.columns else []
    df_filtered = df_sim.dropna(subset=requested_features + target_col)
    dropped_count = len(raw_df) - len(df_filtered)
    print(f"  • Rows before dropna: {len(raw_df)}")
    print(f"  • Rows after dropna:  {len(df_filtered)}")
    print(f"  • Dropped rows:       {dropped_count} ({(dropped_count / len(raw_df) * 100.0) if len(raw_df) > 0 else 0.0:.1f}%)")

    print("\nMissing values breakdown in raw parquet file:")
    for feat in requested_features:
        if feat in raw_df.columns:
            null_cnt = int(raw_df[feat].isna().sum())
            null_pct = (null_cnt / len(raw_df) * 100.0) if len(raw_df) > 0 else 0.0
            print(f"  • {feat:<20}: {null_cnt} missing ({null_pct:.1f}%)")
        else:
            print(f"  • {feat:<20}: COLUMN NOT FOUND in parquet (missing 100.0%)")

    # Additional contextual diagnostic: What numeric features ARE currently in parquet?
    print("\n" + "-" * 70)
    print("ADDITIONAL CONTEXT: Features actually present in ml_training_data_v2.parquet:")
    print("-" * 70)
    for c in raw_df.columns:
        if c in ("target", "date", "ticker", "backtest_date", "outcome", "catalyst_type", "direction", "asset_tier"):
            continue
        if pd.api.types.is_numeric_dtype(raw_df[c]):
            u_cnt = int(raw_df[c].nunique())
            c_std = float(raw_df[c].std()) if len(raw_df) > 1 else 0.0
            v_counts = raw_df[c].value_counts(dropna=False)
            top_p = (v_counts.iloc[0] / len(raw_df)) * 100.0
            corr_val = float(raw_df[c].corr(raw_df["target"])) if "target" in raw_df.columns and c_std > 1e-6 else 0.0
            status_label = "REAL SIGNAL" if (top_p <= 80.0 and c_std >= 1e-4 and u_cnt > 1) else "LIKELY PLACEHOLDER"
            print(f"  * {c:<20} | {status_label:<18} | Unique: {u_cnt:<5} | Std: {c_std:<7.3f} | Corr with target: {corr_val:+.4f}")
        else:
            u_cnt = int(raw_df[c].nunique())
            print(f"  * {c:<20} | {'NON-NUMERIC METADATA':<18} | Unique: {u_cnt:<5} | (string/categorical)")

    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
