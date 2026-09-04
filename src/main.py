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
from src.strategy.momentum_movers import MomentumMoversStrategy
from src.ai.blind_sentiment import BlindSentimentAnalyzer
from src.ai.xgb_micro_v2 import XGBMicroSentinelV2
from src.execution.risk_manager import RiskManager
from src.alerts import AlertGateway

from src.data.dynamic_scanner import DynamicTickerScanner
from src.data.market_gainer_discovery import MarketGainerDiscovery, EXPANDED_UNIVERSE

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TRADIER_TOKEN = os.getenv("TRADIER_ACCESS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def validate_environment():
    """Ensure critical environment variables are present before booting."""
    missing = []
    if not TRADIER_TOKEN: missing.append("TRADIER_ACCESS_TOKEN")
    # Either provider will do; both speak the same protocol.
    if not (GROQ_API_KEY or OPENROUTER_API_KEY):
        missing.append("GROQ_API_KEY or OPENROUTER_API_KEY")
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
        self.gainer_discovery = MarketGainerDiscovery(TRADIER_TOKEN)
        self.momentum_movers = MomentumMoversStrategy()
        self._bar_aggregators = {}
        
        # Initialize market stream with expanded liquid universe
        initial_symbols = EXPANDED_UNIVERSE[:120]
        self.market_stream = TradierMarketStream(TRADIER_TOKEN, initial_symbols, self.event_queue)
        self.orb_strategy = ORBStrategy(orb_minutes=15)
        self.donchian_strategy = DonchianSwingStrategy(TRADIER_TOKEN)
        self.extended_scanner = ExtendedHoursScanner()
        # Resolved from the environment: Groq when GROQ_API_KEY is set, else OpenRouter.
        self.sentiment_analyzer = BlindSentimentAnalyzer()
        self.xgb_model = XGBMicroSentinelV2()
        self.risk_manager = RiskManager()
        from src.execution.risk_budget import RiskBudget
        # Hard loss ceilings. Off until the trader sets them; see risk_budget.py.
        self.risk_budget = RiskBudget()
        self.alerts = AlertGateway(redis_client)
        # The bot cannot see Robinhood, so /took and /closed are how it learns what is
        # actually held. Only confirmed positions receive exit alerts.
        self.alerts.attach_position_ledger(self.risk_manager)
        self.alerts.risk_budget = self.risk_budget
        self._shared_http_session = None  # Lazy-initialized shared aiohttp session
        
        from src.execution.options_pricer import OptionsPricer
        from src.data.news_fetcher import NewsFetcher
        from src.execution.signal_pipeline import SignalPipeline
        self.options_pricer = OptionsPricer(TRADIER_TOKEN)
        self.news_fetcher = NewsFetcher()
        self._last_expiration_check_hour = -1
        # Tickers already warned about a missing daily ATR. The warning below sits in the
        # per-tick path, and atr_cache is empty until the hourly Donchian scan completes,
        # so without this it fires several times a second per position.
        self._atr_warned = set()
        self.signal_pipeline = SignalPipeline(
            self.news_fetcher,
            self.sentiment_analyzer,
            self.xgb_model,
            self.risk_manager,
            self.options_pricer,
            self.alerts,
            self.donchian_strategy,
            self.risk_budget
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
        """Periodically polls Yahoo Finance + Tradier to discover market-wide top gainers & volume runners."""
        log.info("🚀 [MULTI-API DISCOVERY] Starting automated market-wide surge discovery loop...")
        while True:
            try:
                # Concurrent discovery across Yahoo Finance screeners & Tradier bulk quotes
                active_symbols = await self.gainer_discovery.discover_market_movers(max_symbols=150)
                if active_symbols:
                    # Pin every open position into the subscription. Discovery returns
                    # today's movers; a ticker we are holding may not be one of them, and
                    # dropping it from the stream leaves the position with no trailing
                    # stop, no take-profit and no price at all for the life of the option.
                    held = [t for t in self.risk_manager.active_positions if t not in active_symbols]
                    if held:
                        log.info(f"📌 [STREAM] Pinning {len(held)} held position(s) into the subscription: {', '.join(sorted(held))}")
                    await self.market_stream.update_symbols(list(active_symbols) + held)
            except Exception as e:
                log.error(f"Multi-API discovery loop error: {e}")
            await asyncio.sleep(300) # Re-scan every 5 minutes
            
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
                    self.momentum_movers.reset_daily_state()
                    self._bar_aggregators.clear()
                    self.extended_scanner.reset_daily()
                    self.risk_manager.reset_daily(current_date=today.strftime("%Y-%m-%d"))
                    self._atr_warned.clear()
                    self._last_reset_date = today
                    log.info(f"🔄 [DAILY RESET] Strategy states and memory cleared for {today}. Fresh VWAP/EMA/ORB will build from 9:30.")
            except Exception as e:
                log.error(f"Daily reset loop error: {e}")
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def run_hourly_position_sweep_loop(self):
        """Warns hourly about tracked positions whose option is about to expire, and about
        positions that have stalled or lost premium.

        This also dispatched an hourly market-pulse card. That card was breadth the trader
        could read off any chart, it said nothing about the positions actually held, and it
        spent eight of the day's Telegram sends -- against a limit of roughly one message
        per second per chat, which had already cost a full session of alerts."""
        log.info("⏱️ [POSITION SWEEP] Starting hourly position sweep loop...")
        while True:
            try:
                now = pd.Timestamp.now(tz="US/Eastern")
                # Expiration sweep: once per hour during market hours, flag any tracked
                # position whose option is about to expire and was never marked closed.
                if (9 <= now.hour <= 16) and self._last_expiration_check_hour != now.hour:
                    self._last_expiration_check_hour = now.hour
                    await self._dispatch_expiration_warnings(now)
                    await self._dispatch_health_alerts(now)
            except Exception as e:
                log.error(f"Hourly position sweep loop error: {e}")
            await asyncio.sleep(30)

    async def _dispatch_exit_alert(self, ticker: str, pos: dict, outcome: dict) -> None:
        """Tells the trader to act. Entry alerts were always dispatched; exits were only
        written to a CSV, which meant the bot said when to get in and never when to get
        out -- the failure the whole exit-discipline effort exists to fix."""
        status = outcome["status"]
        price = outcome.get("exit_price", 0.0)
        stock_pnl = outcome.get("pnl")
        opt = outcome.get("estimated_option_pnl_pct")
        opt_txt = f"\nEstimated option P&L: <b>{opt*100:+.0f}%</b> <i>(delta+theta approximation)</i>" if opt is not None else ""
        stock_txt = f"{stock_pnl*100:+.2f}%" if stock_pnl is not None else "n/a"

        if status == "TP_HIT_TRAILING":
            trail = outcome.get("trailing_stop")
            body = (f"🎯 <b>TARGET REACHED — {ticker}</b> at <code>${price:.2f}</code> ({stock_txt})"
                    f"{opt_txt}\n\n"
                    f"Runner mode is on, so this is <b>not</b> a close signal. The position now "
                    f"trails at <code>${trail:.2f}</code> and only that stop will end it.\n"
                    f"Consider taking part of it here and letting the rest run.")
        elif status == "TP_HIT":
            body = (f"🎯 <b>TAKE PROFIT — {ticker}</b> at <code>${price:.2f}</code> ({stock_txt})"
                    f"{opt_txt}\n\nTarget reached. Close it.")
        elif status == "RUNNER_STOPPED":
            body = (f"🏁 <b>RUNNER STOPPED — {ticker}</b> at <code>${price:.2f}</code> ({stock_txt})"
                    f"{opt_txt}\n\nThe trail finally caught it after the target. Close it.")
        elif status == "STOPPED_OUT":
            body = (f"🛑 <b>STOPPED OUT — {ticker}</b> at <code>${price:.2f}</code> ({stock_txt})"
                    f"{opt_txt}\n\nTrailing stop hit. Close it.")
        else:  # INVALIDATED
            body = (f"⚠️ <b>THESIS INVALIDATED — {ticker}</b> at <code>${price:.2f}</code> ({stock_txt})"
                    f"{opt_txt}\n\nPrice gave back the level the setup was built on. Close it.")

        body += f"\n\n<i>Reply /closed {ticker} &lt;fill&gt; to record what you actually got.</i>"
        try:
            if not await self.alerts.dispatch_informational(body):
                log.error(f"[{ticker}] EXIT alert dispatch FAILED ({status}). "
                          f"The position may be closed with no notification sent.")
            else:
                log.info(f"[{ticker}] Exit alert dispatched: {status}.")
        except Exception as e:
            log.error(f"[{ticker}] Exit alert dispatch raised ({status}): {e}")

    async def _dispatch_health_alerts(self, now):
        """Alerts on the two exit failures this account's own history shows: holding past
        the point the edge inverts, and riding a loser down instead of cutting it."""
        try:
            alerts = self.risk_manager.check_position_health(now.strftime("%Y-%m-%d"))
        except Exception as e:
            log.error(f"Position health sweep failed: {e}")
            return

        for a in alerts:
            ticker, reason = a["ticker"], a["reason"]
            flag = "time_stop_warned" if reason == "TIME_STOP" else "premium_stop_warned"
            if ticker not in self.risk_manager.active_positions:
                continue

            est = a.get("est_option_pnl_pct")
            est_txt = (f" Estimated premium {est*100:+.0f}% (delta+theta approximation)."
                       if est is not None else " No recent tick, so the position could not be valued.")
            if reason == "TIME_STOP":
                msg = (f"⏳ TIME STOP: {ticker} has been open {a['days_held']} day(s).{est_txt}\n"
                       f"Past day {self.risk_manager.max_hold_days} this account's win rate historically "
                       f"falls from ~50% to 27% and average return goes from positive to -20.7%. "
                       f"Decide now: close it or accept you are outside your own edge.")
            else:
                msg = (f"🛑 PREMIUM STOP: {ticker} {a['detail']}.{est_txt}\n"
                       f"69 past contracts were only sold after losing 50-99% of premium. "
                       f"This is the point where those became unrecoverable.")

            try:
                sent = await self.alerts.dispatch_informational(msg)
            except Exception as e:
                log.error(f"[{ticker}] {reason} dispatch raised: {e}")
                sent = False

            if sent:
                log.warning(f"[{ticker}] {reason} alert dispatched ({a['detail']}).")
            else:
                self.risk_manager.requeue_warning(ticker, flag)
                log.error(f"[{ticker}] {reason} dispatch FAILED; re-queued for the next sweep.")

    async def _dispatch_expiration_warnings(self, now):
        """Sends one Telegram warning per position nearing expiration.

        check_expiration_warnings() consumes a position's expiration_warned flag when it
        hands the warning back, so a send that fails here MUST return the flag -- otherwise
        one Telegram blip silently swallows the only reminder that position will ever get,
        which is exactly the failure this feature exists to prevent.
        """
        try:
            pending = self.risk_manager.check_expiration_warnings(now.strftime("%Y-%m-%d"))
        except Exception as e:
            log.error(f"Expiration sweep failed: {e}")
            return

        for warning in pending:
            ticker = warning["ticker"]
            days_left = warning["days_to_expiration"]

            pos = self.risk_manager.active_positions.get(ticker)
            if pos is None:
                # Closed between the sweep and this dispatch; the alert text would be false.
                log.info(f"[{ticker}] Expiration warning dropped: position closed before dispatch.")
                continue

            try:
                sent = await self.alerts.dispatch_informational(
                    f"⏰ EXPIRATION WARNING: {ticker} option expires in "
                    f"{days_left} day(s). This position has not been marked closed. "
                    f"24 previous contracts were left to expire worthless for a combined $6,008 loss -- "
                    f"don't let this be another one."
                )
            except Exception as e:
                log.error(f"[{ticker}] Expiration warning dispatch raised: {e}")
                sent = False

            if sent:
                log.warning(f"⏰ [EXPIRATION] {ticker} expires in {days_left} day(s) and is still open. Warning dispatched.")
            else:
                # Hand the warning back so the next hourly sweep retries it.
                self.risk_manager.requeue_warning(ticker, "expiration_warned")
                log.error(
                    f"⏰ [EXPIRATION] Telegram dispatch FAILED for {ticker} "
                    f"({days_left} day(s) to expiry). Warning re-queued for the next hourly sweep."
                )

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
                        pos = self.risk_manager.active_positions[ticker]
                        catalyst_type = pos.get("catalyst_type", "Standard Breakout")

                        cached_atr = None
                        if hasattr(self, "donchian_strategy") and hasattr(self.donchian_strategy, "atr_cache"):
                            cached_atr = self.donchian_strategy.atr_cache.get(ticker)
                        
                        if cached_atr and not pd.isna(cached_atr):
                            current_atr = cached_atr
                        else:
                            # Same reasoning as the pipeline: a flat dollar ATR makes the
                            # trailing stop far too tight on cheap stocks and far too loose
                            # on expensive ones.
                            from src.execution.signal_pipeline import FALLBACK_ATR_PCT
                            current_atr = price * FALLBACK_ATR_PCT
                            if ticker not in self._atr_warned:
                                self._atr_warned.add(ticker)
                                log.warning(
                                    f"[{ticker}] No cached daily ATR for the trailing stop; "
                                    f"using {FALLBACK_ATR_PCT*100:.1f}% of price "
                                    f"(ATR={current_atr:.2f}) until the hourly Donchian scan "
                                    f"caches one. Logged once per ticker per day."
                                )
                            
                        direction = self.risk_manager.active_positions[ticker]["direction"]
                        outcome = self.risk_manager.update_trailing_stop(ticker, price, current_atr, direction)
                        
                        if outcome["status"] == "TP_HIT_TRAILING":
                            # Target reached but the position stays open on a tighter
                            # trail, so the move is allowed to keep going.
                            asyncio.create_task(self._dispatch_exit_alert(ticker, pos, outcome))

                        elif outcome["status"] in ("STOPPED_OUT", "TP_HIT", "INVALIDATED", "RUNNER_STOPPED"):
                            confirmed = bool(pos.get("confirmed"))
                            self.risk_manager.close_trade(
                                ticker=ticker,
                                outcome_status=outcome["status"],
                                exit_price=outcome["exit_price"],
                                pnl_pct=outcome["pnl"],
                                estimated_option_pnl_pct=outcome.get("estimated_option_pnl_pct")
                            )
                            self.alerts.record_outcome(catalyst_type=catalyst_type, outcome_status=outcome["status"])
                            # An alerter that never signals the exit is half a tool. Only
                            # for positions actually held -- see the /took ledger.
                            if confirmed:
                                asyncio.create_task(self._dispatch_exit_alert(ticker, pos, outcome))

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

                    # 1.6 Aggregate into 1-minute bars for Momentum Movers Strategy
                    if event.get("type") == "trade" and actual_volume > 0:
                        cur_min = now.minute
                        agg = self._bar_aggregators.get(ticker)
                        if agg is None or agg["minute"] != cur_min:
                            if agg is not None:
                                mover_signal = self.momentum_movers.on_bar(agg)
                                if mover_signal and not signal:
                                    signal = mover_signal
                            self._bar_aggregators[ticker] = {
                                "ticker": ticker,
                                "open": price,
                                "high": price,
                                "low": price,
                                "close": price,
                                "volume": actual_volume,
                                "minute": cur_min,
                                "timestamp": now
                            }
                        else:
                            agg["high"] = max(agg["high"], price)
                            agg["low"] = min(agg["low"], price)
                            agg["close"] = price
                            agg["volume"] += actual_volume
                    
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

    # Restore open option positions from the previous run. Without this a restart
    # silently abandons every tracked contract, which is how positions ended up
    # expiring worthless with nobody watching them.
    try:
        today_et = pd.Timestamp.now(tz="US/Eastern").strftime("%Y-%m-%d")
        restored = app.state.engine.risk_manager.load_positions(current_date=today_et)
        if restored:
            log.info(f"[STATE] Restored {restored} open position(s) from the previous run.")
    except Exception as e:
        log.error(f"[STATE] Failed to restore persisted positions: {e}")
    
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
        msg = "⚠️ XGBoost Sentinel V2 not loaded — no model file or training data found.\nAll signals will show LOW conviction by default. Run run_pipeline.py to generate training data, then retrain before relying on conviction scores."
        log.warning(msg)
        asyncio.create_task(app.state.engine.alerts.dispatch_informational(msg))
    
    app.state.extended_task = asyncio.create_task(app.state.engine.run_extended_hours_scanner_loop())
    app.state.scanner_task = asyncio.create_task(app.state.engine.run_dynamic_scanner_loop())
    app.state.donchian_task = asyncio.create_task(app.state.engine.run_donchian_scanner_loop())
    app.state.position_sweep_task = asyncio.create_task(app.state.engine.run_hourly_position_sweep_loop())
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
    app.state.position_sweep_task.cancel()
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
