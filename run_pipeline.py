import os
import sys
import asyncio
from dotenv import load_dotenv

root_env = os.path.abspath(".env")
load_dotenv(dotenv_path=root_env)
sys.path.insert(0, os.getcwd())

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger("rallyhunter.run_pipeline")

async def run_pipeline():
    log.info("=== STARTING FULL BACKTEST & RETRAINING PIPELINE ===")
    
    # 1. Download Historical 1-Minute Data
    log.info("Step 1/3: Downloading 1-minute historical intraday data for Major Indexes & Mag 7 tickers...")
    from src.data.historical_downloader import main as download_main
    await download_main()
    
    # 2. Run Strategy Simulation
    log.info("Step 2/3: Simulating 4-timeframe strategy across historical bars...")
    import redis.asyncio as aioredis
    from src.main import IntradayEngine
    from src.backtester_v2 import IntradayBacktesterV2
    
    redis = await aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    engine = IntradayEngine(redis)
    backtester = IntradayBacktesterV2(engine)
    
    data_dir = os.path.abspath("data/intraday/")
    await backtester.run_simulation(data_dir)
    await redis.aclose()
    
    # 3. Retrain XGBoost Model
    log.info("Step 3/3: Retraining XGBoost Sentinel ML Model...")
    from src.ai.xgb_micro_v2 import XGBMicroSentinelV2
    model = XGBMicroSentinelV2()
    out_parquet = os.path.abspath("data/ml_training_data_v2.parquet")
    if os.path.exists(out_parquet):
        model.train(out_parquet)
        log.info("🎉 SUCCESS: Backtest complete & XGBoost model retrained!")
    else:
        log.warning("Parquet dataset file not found.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
