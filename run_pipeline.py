import os
import sys
import asyncio
import pandas as pd
from typing import Optional
from dotenv import load_dotenv

if sys.platform == "win32":
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

root_env = os.path.abspath(".env")
load_dotenv(dotenv_path=root_env)
sys.path.insert(0, os.getcwd())

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger("rallyhunter.run_pipeline")

from src.backtest.stats_reader import BacktestStatsReader

def prepare_ml_training_data(csv_path: Optional[str] = None, output_parquet: str = "data/ml_training_data_v2.parquet") -> Optional[str]:
    """
    Finds the most recent backtest CSV, maps labels and features, audits feature realness,
    and exports data/ml_training_data_v2.parquet for XGBoost model retraining.
    """
    if not csv_path:
        reader = BacktestStatsReader()
        if not reader.reload_latest_results() or not reader._last_loaded_file:
            log.error("No backtest CSV found in backtest_results/ directory.")
            return None
        csv_path = reader._last_loaded_file

    log.info(f"Bridging backtest results from {csv_path} to ML training parquet...")
    df = pd.read_csv(csv_path)
    if df.empty:
        log.warning(f"Backtest CSV {csv_path} is empty.")
        return None

    # Map target: WIN -> 1, everything else (LOSS, EXPIRED, INVALIDATED) -> 0
    if "outcome" in df.columns:
        df["target"] = (df["outcome"].astype(str).str.upper() == "WIN").astype(int)
    else:
        log.error("CSV missing 'outcome' column required for ML target.")
        return None

    # Map date column
    if "backtest_date" in df.columns:
        df["date"] = pd.to_datetime(df["backtest_date"])
    elif "entry_date" in df.columns:
        df["date"] = pd.to_datetime(df["entry_date"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.Timestamp.now()

    # The 9 real features xgb_micro_v2.py expects for 14-21 DTE options
    real_features = []
    placeholder_features = []

    # 1. relative_volume (mapped from z_vol if relative_volume not present)
    if "relative_volume" in df.columns:
        real_features.append("relative_volume")
    elif "z_vol" in df.columns:
        df["relative_volume"] = df["z_vol"].astype(float)
        real_features.append("relative_volume (mapped from 'z_vol')")
    else:
        df["relative_volume"] = 1.5
        placeholder_features.append("relative_volume (default 1.5)")

    # 2. direction_code
    if "direction_code" in df.columns:
        real_features.append("direction_code")
    elif "direction" in df.columns:
        df["direction_code"] = (df["direction"].astype(str).str.upper() == "LONG").astype(int)
        real_features.append("direction_code (mapped from 'direction')")
    else:
        df["direction_code"] = 1
        placeholder_features.append("direction_code (default 1)")

    # 3. Core quantitative features
    feature_defaults = {
        "rsi_14": 55.0,
        "chop_14": 45.0,
        "expected_move_pct": 4.5,
        "hist_vol_20": 0.35,
        "sma20_ratio": 1.02,
        "sma_spread": 0.03,
        "breakout_pct": 0.015
    }

    for feat, default_val in feature_defaults.items():
        if feat in df.columns and df[feat].notna().any():
            real_features.append(feat)
        else:
            df[feat] = default_val
            placeholder_features.append(f"{feat} (filled with fallback {default_val})")

    # Clean audit printout to console
    print("\n" + "=" * 68)
    print("  --- ML FEATURE BRIDGING & DATA HEALTH AUDIT ---")
    print("=" * 68)
    print(f"• Source Backtest File: {os.path.basename(csv_path)}")
    print(f"• Total Rows Converted: {len(df)} (Wins: {df['target'].sum()}, Losses/Other: {len(df) - df['target'].sum()})")
    print(f"\n[+] Real Signal Features ({len(real_features)}):")
    for rf in real_features:
        print(f"   [REAL SIGNAL]        • {rf}")
    print(f"\n[!] Placeholder Features ({len(placeholder_features)}) [NO REAL SIGNAL]:")
    for pf in placeholder_features:
        print(f"   [PLACEHOLDER FILL]   • {pf}")
    print("=" * 68 + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    log.info(f"✅ Successfully wrote {len(df)} samples to {output_parquet}")
    return output_parquet

async def run_pipeline():
    log.info("=== STARTING FULL BACKTEST & RETRAINING PIPELINE ===")
    
    # 1. Download Historical 1-Minute Data
    log.info("Step 1/4: Downloading 1-minute historical intraday data for Major Indexes & Mag 7 tickers...")
    from src.data.historical_downloader import main as download_main
    await download_main()
    
    # 2. Run Strategy Simulation
    log.info("Step 2/4: Simulating 4-timeframe & options pricing strategy across historical bars...")
    from src.backtest.engine import BacktestEngine
    from src.data.dynamic_scanner import CANDIDATE_POOL
    
    token = os.getenv("TRADIER_ACCESS_TOKEN")
    engine = BacktestEngine(token)
    await engine.run_backtest(CANDIDATE_POOL, max_holding_days=10, lookback_days=730)
    
    # 3. Bridge Backtest CSV to Parquet Training Data
    log.info("Step 3/4: Converting latest backtest CSV to ML training parquet format...")
    out_parquet = os.path.abspath("data/ml_training_data_v2.parquet")
    parquet_path = prepare_ml_training_data(output_parquet=out_parquet)
    
    # 4. Retrain XGBoost Model
    log.info("Step 4/4: Retraining XGBoost Sentinel ML Model...")
    from src.ai.xgb_micro_v2 import XGBMicroSentinelV2
    model = XGBMicroSentinelV2()
    if parquet_path and os.path.exists(parquet_path):
        model.train(parquet_path)
        log.info("🎉 SUCCESS: Backtest complete & XGBoost model retrained!")
    else:
        log.warning("Parquet dataset file not found. Skipping training.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bridge-only":
        prepare_ml_training_data()
    else:
        asyncio.run(run_pipeline())
