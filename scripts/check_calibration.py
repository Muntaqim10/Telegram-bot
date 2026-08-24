import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    # 1. Load data and recreate the validation split exact logic
    training_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ml_training_data_v2.parquet"))
    df = pd.read_parquet(training_data_path)
    
    features = [
        "entry_time_minute", "relative_volume", "vwap_ratio", 
        "ema9_ratio", "ema_trend", "ema_trend_5m", 
        "spy_correlation", "hod_ratio", "lod_ratio"
    ]
    
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna(subset=features + ["target"])
    
    from src.ai.xgb_micro_v2 import get_train_val_split
    _, _, X_val, y_val = get_train_val_split(df, features, "target")
        
    # 2. Load the candidate model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models/xgb_micro_v2_candidate.json"))
    model = xgb.Booster()
    model.load_model(model_path)
    
    # 3. Generate predictions
    dval = xgb.DMatrix(X_val)
    preds = model.predict(dval)
    
    val_df = pd.DataFrame({
        "pred": preds,
        "actual": y_val.values
    })
    
    # 4 & 5. Bucket predictions
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    val_df["bucket"] = pd.cut(val_df["pred"], bins=bins, labels=labels, right=True, include_lowest=True)
    
    print("\n" + "="*60)
    print("CALIBRATION BUCKET TABLE")
    print("="*60)
    print(f"{'Bucket':<12} | {'Count':<6} | {'Mean Pred':<10} | {'Realized':<10} | {'Gap (Err)':<10}")
    print("-" * 60)
    
    for label in labels:
        b_df = val_df[val_df["bucket"] == label]
        count = len(b_df)
        if count == 0:
            print(f"{label:<12} | {count:<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
            continue
            
        mean_pred = b_df["pred"].mean()
        actual_win_rate = b_df["actual"].mean()
        gap = mean_pred - actual_win_rate
        
        print(f"{label:<12} | {count:<6} | {mean_pred:<10.3f} | {actual_win_rate:<10.3f} | {gap:<10.3f}")
        
    # 6. Overall Brier Score
    brier_score = np.mean((val_df["pred"] - val_df["actual"]) ** 2)
    print("-" * 60)
    print(f"Overall Brier Score (MSE): {brier_score:.4f}\n")
    
    # 7. Tier breakdown
    def get_tier(p):
        if p >= 0.75: return "HIGH (>=0.75)"
        if p >= 0.55: return "MEDIUM (0.55-0.75)"
        return "LOW (<0.55)"
        
    val_df["tier"] = val_df["pred"].apply(get_tier)
    
    print("="*60)
    print("THREE-TIER ALERTS BREAKDOWN")
    print("="*60)
    
    tiers = ["HIGH (>=0.75)", "MEDIUM (0.55-0.75)", "LOW (<0.55)"]
    for t in tiers:
        t_df = val_df[val_df["tier"] == t]
        count = len(t_df)
        if count == 0:
            print(f"{t:<20} | Count: {count:<4} | Realized Win Rate: N/A")
        else:
            actual_wr = t_df["actual"].mean()
            print(f"{t:<20} | Count: {count:<4} | Realized Win Rate: {actual_wr:.3f}")
            
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
