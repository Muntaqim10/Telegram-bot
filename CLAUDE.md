# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the engine (FastAPI + background tasks, uvicorn on 0.0.0.0:8000)
uv run python src/main.py          # or: run_bot.bat

# Tests
python -m pytest tests/ -q                       # all of them
python -m pytest tests/test_runner_mode.py -q    # one file
python -m pytest tests/ -k "premium_stop" -q     # one behaviour

# Lint. The configured rule set (E,F,I,B,C4,UP,S) has a ~684-error backlog nobody
# has triaged, so day-to-day the convention is real-bug rules only:
python -m ruff check src/ scripts/ --select F821,F841
```

**Run the whole suite before committing** — CI does, and a change to the position dict
routinely breaks a module three files away because the suites cover overlapping
behaviour. `tests/conftest.py` holds the shared fixtures; two of them exist specifically
so no test can reach the real `data/active_positions.json` or `data/rallyhunter.db`.

`tests/test_model_honesty.py` skips its model-dependent cases when
`data/models/xgb_micro_v2.json` is absent — `data/` is gitignored, so that is the normal
state in CI.

Analysis and diagnostics (all read-only unless noted):

```bash
python scripts/import_robinhood.py <statement.csv>   # writes real_fills; idempotent
python scripts/check_live_calibration.py             # predictions vs real money
python scripts/check_feature_signal.py               # is there anything to learn?
python scripts/build_training_set.py --bars <bars.json> --out data/real_training_set.parquet
```

**Windows environment.** Bare `python` / `python3` resolve to the Windows Store stub and
fail — use `./.venv/Scripts/python.exe` (3.12) or `py -3`. PowerShell 5.1 has no `&&`,
`mkdir -p`, or `printf`; use Git Bash for those. Logs and alerts contain emoji, so prefix
anything that prints them with `PYTHONIOENCODING=utf-8` or it dies on cp1252.

## Architecture

**This bot alerts. It does not trade.** Execution happens by hand on Robinhood, so the
bot cannot see the account. Everything it records is either a *suggestion* (an alert) or
a *simulation* (a trailing stop run on stock ticks). Most non-obvious behaviour follows
from this.

### Signal flow

`stream_market` (Tradier websocket) → strategy (`orb_intraday`, `donchian_daily`,
`momentum_movers`, `extended_hours_scanner`) → `signal_pipeline.process_signal()` →
`AlertGateway` → Telegram.

`process_signal()` is a gauntlet of ~13 early-return gates and makes **nine awaited
network calls**. Every gate that returns early **must** call
`risk_manager.remove_position(ticker)` first — see the position lifecycle below.

Measured over 589 real signals: the Velocity Gate (273 blocks) and Exhaustion Gate (99)
do most of the filtering with plain arithmetic. The XGBoost fakeout filter has blocked
**zero**. The DeepSeek synthesis call gates nothing — it only writes `catalyst` and
`ai_thesis`, which are display text.

### Position lifecycle — easy to get wrong

1. `add_position()` is **speculative**. The pipeline registers a position *before* it
   knows whether the alert will survive the remaining gates.
2. A gate that suppresses the alert calls `remove_position()` — no outcome logged.
   `close_trade()` is only for positions that were genuinely tracked and reached an exit.
3. `attach_option_pricing()` is where `option_expiration` lands, because the pipeline
   registers the position before it resolves the option chain.
4. A position is **not a holding** until the trader confirms it with `/took`. Only
   confirmed positions receive expiration, time-stop and premium-stop warnings —
   otherwise the bot nags about trades that were never made.
5. `reset_daily(current_date)` keeps confirmed positions whose option has not expired and
   drops everything else. Called without a date it clears everything (legacy behaviour).
6. State persists to `data/active_positions.json` (atomic replace) and reloads at boot.

Held tickers are pinned into the market-stream subscription in
`run_dynamic_scanner_loop`; without that, discovery rotation silently unsubscribes an
open position and it goes unpriced for the life of the contract.

### Dates

All dates come from `RiskManager.market_today()` (US/Eastern) and use `%Y-%m-%d`. Never
use `datetime.now()` for anything that will be compared against another date — the host
may run UTC, and entry dates recorded from an evening signal will land a day ahead.

### Three sources of truth

| Source | What it is | Trust |
|---|---|---|
| `real_fills` | Robinhood statement + `/closed` fills | Money |
| `trade_log` / `trade_log_archive` | alerts, and simulated trailing-stop outcomes | Counterfactual |
| `data/ml_training_data_v2.parquet` | backtest simulation | Furthest from reality |

`real_fills.matched_alert_id` must always be read alongside `matched_alert_table` — the
two alert tables autoincrement separately and 61 of their ids collide.

### Decision inputs are capture-only

Greeks, IV and sentiment exist only at the moment an alert is built. Historical option
chains are not retrievable and re-scoring old headlines leaks hindsight, so anything not
written to `trade_log` at dispatch is **gone permanently**. If you add an input the model
might use, journal it in `AlertGateway.dispatch_high_conviction` and add the column in
`database._init_db_sync`.

### Model honesty

The sentinel is scored only when the strategy actually supplied `sma_spread`,
`sma20_ratio` and `rsi_14`. Only `donchian_daily` computes them; intraday strategies
borrow them from `donchian_strategy.feature_cache`, and when neither is available the
alert is marked `⚪ UNSCORED` and shows no percentage. A 250-point probe at model load
disables the conviction tiers outright if the score cannot vary. Do not reintroduce
default feature values — scoring constants is what made every intraday alert land on the
same two numbers.

Out-of-sample AUC is 0.50–0.53 on backtest data and 0.5617 ± 0.0738 on 272 real fills.
Treat any claim of predictive power as unproven until `check_live_calibration.py` reports
at least ~30 alert-driven fills.

### Alert delivery

A warning's "already sent" flag is consumed when it is *generated*, not when it is sent.
`dispatch_informational` returns `False` on failure and never raises, so every dispatch
site must check the return value and call `requeue_warning()` — otherwise one Telegram
blip silently swallows the only warning that position will ever produce.

## Configuration

Behaviour-changing environment variables, all opt-in unless noted:

- `OPTION_MIN_DTE` / `OPTION_MAX_DTE` — contract target window (default 14/21)
- `MAX_PREMIUM_DOLLARS` — affordability cap (default `max(15, strike × 0.10)`)
- `HOLD_DAYS` — scales the take-profit target (default 3)
- `RUNNER_MODE` / `RUNNER_TRAIL_ATR` — let a winner run past the target instead of closing
- `RISK_MAX_LOSS_PER_WEEK` / `_PER_MONTH` / `RISK_MAX_PREMIUM_PER_TRADE` / `RISK_MAX_OPEN_POSITIONS`
- `RETRAIN_AT` — retrain timestamp used to split calibration results

Required at boot (`validate_environment()` exits without them): `TRADIER_ACCESS_TOKEN`,
`OPENROUTER_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Known dead weight

Eleven `TradeSignal` fields are assigned and never read: `volume`, `timeframe_target`,
`cycle_type`, `timeframe_matrix`, `earnings_risk`, `gex_confidence`, `otm_runner`,
`otm_qualified`, `is_early_surge`, `early_surge_desc`, `invalidation_level`. Nothing
imports `src/backtest/engine.py` or `src/data/historical_downloader.py`.

`earnings_risk` is the one worth fixing rather than deleting — it is `True` when earnings
falls inside the option's life, which is the largest IV-crush risk in this strategy, and
the alert never mentions it.

OTM contracts are hard-disabled in `signal_pipeline` (`raw_otm = None`,
`otm_qualified = False`); the bot only ever proposes ~0.75-delta ITM.
