import asyncio
import logging
import os
import redis.asyncio as aioredis
from datetime import datetime
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
config_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/.env"))
load_dotenv(dotenv_path=root_env)
load_dotenv(dotenv_path=config_env)
from pydantic import BaseModel
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from logging.handlers import RotatingFileHandler

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("logs/rallyhunter.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    ]
)
log = logging.getLogger("rallyhunter.main")

from src.data.stream_market import TradierMarketStream
from src.strategy.orb_intraday import ORBStrategy
from src.strategy.donchian_daily import DonchianSwingStrategy
from src.ai.blind_sentiment import BlindSentimentAnalyzer
from src.ai.xgb_micro import XGBMicroSentinel
from src.execution.risk_manager import RiskManager
from src.alerts import AlertGateway

from src.data.dynamic_scanner import DynamicTickerScanner, CANDIDATE_POOL

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TRADIER_TOKEN = os.getenv("TRADIER_ACCESS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class IntradayEngine:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.event_queue = asyncio.Queue()
        self.dynamic_scanner = DynamicTickerScanner(TRADIER_TOKEN)
        # Initialize market stream with dynamic candidate pool
        initial_symbols = CANDIDATE_POOL[:40]
        self.market_stream = TradierMarketStream(TRADIER_TOKEN, initial_symbols, self.event_queue)
        self.orb_strategy = ORBStrategy(orb_minutes=15)
        self.donchian_strategy = DonchianSwingStrategy(TRADIER_TOKEN)
        self.sentiment_analyzer = BlindSentimentAnalyzer(GROQ_API_KEY)
        self.xgb_model = XGBMicroSentinel()
        self.risk_manager = RiskManager()
        self.alerts = AlertGateway(redis_client)
        
        from src.execution.options_pricer import OptionsPricer
        from src.data.news_fetcher import NewsFetcher
        from src.execution.signal_pipeline import SignalPipeline
        self.options_pricer = OptionsPricer(TRADIER_TOKEN)
        self.news_fetcher = NewsFetcher()
        self.signal_pipeline = SignalPipeline(
            self.news_fetcher,
            self.sentiment_analyzer,
            self.xgb_model,
            self.risk_manager,
            self.options_pricer,
            self.alerts
        )

    async def run_dynamic_scanner_loop(self):
        """Periodically scans the broad market for high-volume surge & momentum leaders."""
        log.info("⚡ [DYNAMIC SCANNER] Starting automated broad market discovery loop...")
        while True:
            try:
                active_symbols = await self.dynamic_scanner.get_active_market_movers(max_symbols=50)
                if active_symbols:
                    self.market_stream.update_symbols(active_symbols)
            except Exception as e:
                log.error(f"Dynamic scanner loop error: {e}")
            await asyncio.sleep(900) # Scan every 15 minutes
            
    async def run_donchian_scanner_loop(self):
        """Periodically scans for daily Donchian swing setups (runs hourly)."""
        log.info("📅 [DONCHIAN SCANNER] Starting daily swing scanner loop...")
        while True:
            try:
                active_symbols = self.market_stream.symbols
                if active_symbols:
                    signals = await self.donchian_strategy.scan_universe(active_symbols)
                    for signal in signals:
                        # Pipe Donchian daily signals straight into the signal pipeline
                        # SPY VWAP ratio doesn't apply to daily bars, so we default to 1.0
                        await self.signal_pipeline.process_signal(signal, spy_vwap_ratio=1.0)
            except Exception as e:
                log.error(f"Donchian scanner loop error: {e}")
            await asyncio.sleep(3600) # Scan every hour
        
    async def process_events(self):
        """Main event loop handling ticks from WebSockets."""
        log.info("Intraday Execution Engine Started. Listening for ticks...")
        while True:
            try:
                event = await self.event_queue.get()
                
                # Tradier format: {"type": "trade", "symbol": "AAPL", "price": 150.0, "size": 100, "date": 1690000000000}
                if event.get("type") in ("trade", "quote"):
                    ticker = event.get("symbol")
                    raw_price = event.get("price") or event.get("last")
                    raw_size = event.get("size", 0)
                    
                    if not raw_price or not ticker:
                        continue
                    
                    try:
                        price = float(raw_price)
                        size = float(raw_size) if raw_size is not None else 0.0
                    except (ValueError, TypeError):
                        continue
                        
                    # 0. Update Active Positions in Risk Manager
                    if ticker in self.risk_manager.active_positions:
                        # Fallback ATR if not available yet in state
                        current_atr = 1.5 
                        spy_state = self.orb_strategy.intraday_state.get(ticker, {})
                        if spy_state and "atr" in spy_state:
                            current_atr = spy_state["atr"]
                            
                        direction = self.risk_manager.active_positions[ticker]["direction"]
                        outcome = self.risk_manager.update_trailing_stop(ticker, price, current_atr, direction)
                        
                        if outcome["status"] in ("STOPPED_OUT", "TP_HIT"):
                            self.risk_manager.close_trade(
                                ticker=ticker,
                                outcome_status=outcome["status"],
                                exit_price=outcome["exit_price"],
                                pnl_pct=outcome["pnl"]
                            )

                    # 1. Process Tick in Strategy
                    signal = self.orb_strategy.process_tick(
                        ticker=ticker, 
                        price=price, 
                        volume=size, 
                        timestamp=pd.Timestamp.now(tz="US/Eastern")
                    )
                    
                    if signal:
                        log.info(f"🚨 STRATEGY TRIGGER: {signal['direction']} on {ticker} at {price}")
                        
                        spy_state = self.orb_strategy.intraday_state.get("SPY", {})
                        spy_vwap_ratio = 1.0
                        if spy_state and spy_state.get("vwap"):
                            spy_vwap_ratio = spy_state["last_price"] / spy_state["vwap"]
                            
                        await self.signal_pipeline.process_signal(signal, spy_vwap_ratio)
                        
            except Exception as e:
                log.error(f"Event Loop Error: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[START] RallyHunter_v26 Intraday Engine initializing...")
    app.state.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    
    app.state.engine = IntradayEngine(app.state.redis)
    
    # Pre-flight Type Safety Self-Diagnostic Check
    log.info("[DIAGNOSTIC] Running pre-flight string tick & type safety verification...")
    try:
        test_sig = app.state.engine.orb_strategy.process_tick(
            "SPY", "500.00", "100", pd.Timestamp.now(tz="US/Eastern")
        )
        log.info("[DIAGNOSTIC] ✅ Type Safety Verification: PASSED (String ticks parsed as floats cleanly)")
    except Exception as diag_err:
        log.error(f"[DIAGNOSTIC] ❌ Pre-flight check failed: {diag_err}")

    # Load ML models
    if app.state.engine.xgb_model._is_trained:
        log.info("[ML MODEL] ✅ Trained XGBoost Sentinel model weights loaded into memory.")
    elif os.path.exists("data/ml_training_data.parquet"):
        app.state.engine.xgb_model.train("data/ml_training_data.parquet")
    
    # Start tasks
    app.state.scanner_task = asyncio.create_task(app.state.engine.run_dynamic_scanner_loop())
    app.state.donchian_task = asyncio.create_task(app.state.engine.run_donchian_scanner_loop())
    app.state.stream_task = asyncio.create_task(app.state.engine.market_stream.run())
    app.state.processor_task = asyncio.create_task(app.state.engine.process_events())
    app.state.listener_task = asyncio.create_task(app.state.engine.alerts.start_listener())
    
    yield
    
    log.info("[SHUTDOWN] Terminating tasks...")
    app.state.engine.market_stream.stop()
    app.state.scanner_task.cancel()
    app.state.donchian_task.cancel()
    app.state.stream_task.cancel()
    app.state.processor_task.cancel()
    app.state.listener_task.cancel()
    await app.state.engine.alerts.close()
    await app.state.redis.aclose()

app = FastAPI(title="RallyHunter_v26 Intraday Engine", lifespan=lifespan)

@app.get("/")
@app.get("/health")
async def health_check():
    active_symbols = app.state.engine.market_stream.symbols if hasattr(app.state, "engine") else []
    return {
        "status": "online",
        "service": "RallyHunter Intraday Engine",
        "mode": "Dynamic Broad Market Discovery (No Watchlist)",
        "tracked_tickers_count": len(active_symbols),
        "tracked_tickers": active_symbols
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
