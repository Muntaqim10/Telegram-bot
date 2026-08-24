import asyncio
import logging
import os
import aiohttp
import redis.asyncio as aioredis
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
config_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/.env"))
load_dotenv(dotenv_path=root_env)
load_dotenv(dotenv_path=config_env)
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from logging.handlers import RotatingFileHandler

# Setup Logging
os.makedirs("logs", exist_ok=True)
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
from src.strategy.extended_hours_scanner import ExtendedHoursScanner
from src.ai.blind_sentiment import BlindSentimentAnalyzer
from src.ai.xgb_micro_v2 import XGBMicroSentinelV2
from src.execution.risk_manager import RiskManager
from src.alerts import AlertGateway

from src.data.dynamic_scanner import DynamicTickerScanner, CANDIDATE_POOL

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TRADIER_TOKEN = os.getenv("TRADIER_ACCESS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def validate_environment():
    """Ensure critical environment variables are present before booting."""
    missing = []
    if not TRADIER_TOKEN: missing.append("TRADIER_ACCESS_TOKEN")
    if not OPENROUTER_API_KEY: missing.append("OPENROUTER_API_KEY")
    if not TELEGRAM_BOT_TOKEN: missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID: missing.append("TELEGRAM_CHAT_ID")
    
    if missing:
        log.error(f"CRITICAL BOOT FAILURE: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

# Run validation immediately
validate_environment()

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
        self.extended_scanner = ExtendedHoursScanner()
        self.sentiment_analyzer = BlindSentimentAnalyzer(OPENROUTER_API_KEY)
        self.xgb_model = XGBMicroSentinelV2()
        self.risk_manager = RiskManager()
        self.alerts = AlertGateway(redis_client)
        self._shared_http_session = None  # Lazy-initialized shared aiohttp session
        
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
            self.alerts,
            self.donchian_strategy
        )
        self._last_reset_date = None  # Track the last date we reset ORB state

    async def get_shared_session(self) -> aiohttp.ClientSession:
        """Lazily create and return a shared aiohttp session for all HTTP calls."""
        if self._shared_http_session is None or self._shared_http_session.closed:
            self._shared_http_session = aiohttp.ClientSession()
        return self._shared_http_session

    async def close(self):
        """Clean up shared resources."""
        if self._shared_http_session and not self._shared_http_session.closed:
            await self._shared_http_session.close()
        await self.alerts.close()

    async def run_dynamic_scanner_loop(self):
        """Periodically scans the broad market for high-volume surge & momentum leaders."""
        log.info("⚡ [DYNAMIC SCANNER] Starting automated broad market discovery loop...")
        while True:
            try:
                active_symbols = await self.dynamic_scanner.get_active_market_movers(max_symbols=50)
                if active_symbols:
                    await self.market_stream.update_symbols(active_symbols)
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
                        asyncio.create_task(self._dispatch_signal_task(signal, 1.0))
            except Exception as e:
                log.error(f"Donchian scanner loop error: {e}")
            await asyncio.sleep(3600) # Scan every hour
        
    async def run_extended_hours_scanner_loop(self):
        """Periodically polls the extended hours scanner for alerts."""
        log.info("🌙 [EXTENDED SCANNER] Starting extended hours loop...")
        while True:
            try:
                alerts = self.extended_scanner.evaluate_movers(pd.Timestamp.now(tz="US/Eastern"))
                for alert in alerts:
                    await self.alerts.dispatch_extended_hours(alert)
            except Exception as e:
                log.error(f"Extended scanner loop error: {e}")
            await asyncio.sleep(60)

    async def run_daily_reset_loop(self):
        """Resets ORB intraday state at 9:29 AM ET each trading day to prevent stale VWAP/EMA carry-over."""
        log.info("🔄 [DAILY RESET] Starting daily reset scheduler...")
        while True:
            try:
                now = pd.Timestamp.now(tz="US/Eastern")
                today = now.date()
                # Reset once per day, at or after 9:29 AM, before the ORB window opens
                if now.hour == 9 and now.minute >= 29 and self._last_reset_date != today:
                    self.orb_strategy.reset_daily()
                    self.extended_scanner.reset_daily()
                    self.risk_manager.reset_daily()
                    self._last_reset_date = today
                    log.info(f"🔄 [DAILY RESET] Strategy states and memory cleared for {today}. Fresh VWAP/EMA/ORB will build from 9:30.")
            except Exception as e:
                log.error(f"Daily reset loop error: {e}")
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def _dispatch_signal_task(self, signal: dict, spy_vwap_ratio: float):
        """Background wrapper to process signals without blocking the tick stream loop."""
        ticker = signal["ticker"]
        # Try to acquire the evaluation lock to prevent double entries
        if not self.risk_manager.mark_pending(ticker):
            log.info(f"🚫 BLOCKING {ticker}: Already an active or pending position.")
            return
            
        try:
            session = await self.get_shared_session()
            await self.signal_pipeline.process_signal(signal, spy_vwap_ratio, session=session)
        except Exception as e:
            log.error(f"Error in background signal pipeline for {ticker}: {e}")
        finally:
            # Always release the lock
            self.risk_manager.clear_pending(ticker)

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

                        cached_atr = None
                        if hasattr(self, "donchian_strategy") and hasattr(self.donchian_strategy, "atr_cache"):
                            cached_atr = self.donchian_strategy.atr_cache.get(ticker)
                        
                        if cached_atr and not pd.isna(cached_atr):
                            current_atr = cached_atr
                        else:
                            log.warning(f"[{ticker}] No cached daily ATR available for trailing stop. Falling back to default ATR=1.5")
                            current_atr = 1.5
                            
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
                    now = pd.Timestamp.now(tz="US/Eastern")  # Cache once per tick
                    
                    # VWAP FIX: Only pass volume for actual trades. Quotes inflate VWAP.
                    actual_volume = size if event.get("type") == "trade" else 0.0
                    
                    signal = self.orb_strategy.process_tick(
                        ticker=ticker, 
                        price=price, 
                        volume=actual_volume, 
                        timestamp=now
                    )
                    
                    # 1.5 Process for extended hours movers
                    self.extended_scanner.process_event(event, now)
                    
                    # 2. Priority check for gap movers on open
                    if not signal:
                        signal = self.extended_scanner.consume_priority_signal(ticker, now)
                        if signal:
                            log.info(f"🚨 STRATEGY TRIGGER: Priority Open Eval for Extended Hours Mover {ticker}")
                    
                    if signal:
                        log.info(f"🚨 STRATEGY TRIGGER: {signal['direction']} on {ticker} at {price}")
                        
                        spy_state = self.orb_strategy.intraday_state.get("SPY", {})
                        spy_vwap_ratio = 1.0
                        if spy_state and spy_state.get("vwap"):
                            spy_vwap_ratio = spy_state["last_price"] / spy_state["vwap"]
                            
                        # Dispatch into background task to prevent blocking the tick loop!
                        asyncio.create_task(self._dispatch_signal_task(signal, spy_vwap_ratio))
                        
            except Exception as e:
                log.error(f"Event Loop Error: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[START] RallyHunter_v26 Intraday Engine initializing...")
    app.state.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    
    import src.database as db
    db._init_db_sync()
    
    app.state.engine = IntradayEngine(app.state.redis)
    
    # Pre-flight Type Safety Self-Diagnostic Check
    log.info("[DIAGNOSTIC] Running pre-flight string tick & type safety verification...")
    try:
        app.state.engine.orb_strategy.process_tick(
            "SPY", "500.00", "100", pd.Timestamp.now(tz="US/Eastern")
        )
        log.info("[DIAGNOSTIC] ✅ Type Safety Verification: PASSED (String ticks parsed as floats cleanly)")
    except Exception as diag_err:
        log.error(f"[DIAGNOSTIC] ❌ Pre-flight check failed: {diag_err}")

    # Load ML models
    if app.state.engine.xgb_model._is_trained:
        log.info("[ML MODEL] ✅ Trained XGBoost Sentinel model weights loaded into memory.")
    elif os.path.exists("data/ml_training_data_v2.parquet"):
        app.state.engine.xgb_model.train("data/ml_training_data_v2.parquet")
    else:
        msg = "⚠️ XGBoost Sentinel V2 not loaded — no model file or training data found.\nAll signals will show LOW conviction by default. Run backtester_v2 to generate training data, then retrain before relying on conviction scores."
        log.warning(msg)
        asyncio.create_task(app.state.engine.alerts.dispatch_informational(msg))
    
    app.state.extended_task = asyncio.create_task(app.state.engine.run_extended_hours_scanner_loop())
    app.state.scanner_task = asyncio.create_task(app.state.engine.run_dynamic_scanner_loop())
    app.state.donchian_task = asyncio.create_task(app.state.engine.run_donchian_scanner_loop())
    app.state.daily_reset_task = asyncio.create_task(app.state.engine.run_daily_reset_loop())
    app.state.stream_task = asyncio.create_task(app.state.engine.market_stream.run())
    app.state.processor_task = asyncio.create_task(app.state.engine.process_events())
    app.state.listener_task = asyncio.create_task(app.state.engine.alerts.start_listener())
    
    yield
    
    log.info("[SHUTDOWN] Terminating tasks...")
    app.state.engine.market_stream.stop()
    app.state.extended_task.cancel()
    app.state.scanner_task.cancel()
    app.state.donchian_task.cancel()
    app.state.daily_reset_task.cancel()
    app.state.stream_task.cancel()
    app.state.processor_task.cancel()
    app.state.listener_task.cancel()
    await app.state.engine.close()
    await app.state.redis.aclose()

app = FastAPI(title="RallyHunter_v26 Intraday Engine", lifespan=lifespan)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

@app.get("/")
@app.get("/health")
async def health_check():
    active_symbols = app.state.engine.market_stream.symbols if hasattr(app.state, "engine") else []
    return {
        "status": "online",
        "service": "RallyHunter Intraday Engine",
        "mode": "Dynamic Broad Market Discovery (No Watchlist)",
        "tracked_tickers_count": len(active_symbols),
        "ml_model_active": app.state.engine.xgb_model.is_active if hasattr(app.state, "engine") else False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
