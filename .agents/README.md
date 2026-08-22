# 🚀 RallyHunter_v26 — Parabolic Momentum & Options Detector

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-v7.0%2B-red.svg)](https://redis.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Meta--Classifier-orange.svg)](https://xgboost.readthedocs.io)
[![LLM Scorer](https://img.shields.io/badge/Groq-Llama--3.1--8B-brightgreen.svg)](https://console.groq.com)

RallyHunter_v26 is an institutional-grade, event-driven quantitative scanner designed to identify equities entering high-velocity parabolic breakouts and breakdowns. The engine maps real-time underlying technical breakouts to options chains, filters signals using an **XGBoost Meta-Classifier (Sentinel)**, and validates setups using an **AI Sentiment & Chart Pattern Scorer** (powered by Groq Llama-3.1-8B) before dispatching alerts to Telegram.

---

## 🛠️ System Architecture & Data Flow

RallyHunter uses an asynchronous, non-blocking architecture (via FastAPI and `asyncio`) to ensure sub-millisecond execution times. 

```
                          ┌────────────────────────┐
                          │    Tradier API / WS    │
                          └───────────┬────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ FastAPI Server Daemon                                                    │
│                                                                          │
│  ┌─────────────────────────┐               ┌──────────────────────────┐  │
│  │   Background Scan Loop  ├──────────────►│    Redis State Cache     │  │
│  │   (Every 30s / Async)   │               │  - Indicators & Spreads  │  │
│  └────────────┬────────────┘               │  - Sentiment Cache (15m) │  │
│               │                            │  - Quiet TTL (4 hours)   │  │
│               ▼                            └────────────┬─────────────┘  │
│  ┌─────────────────────────┐                            │                │
│  │   Technical Scanner     │◄───────────────────────────┘                │
│  │  - Volume Z-Score > 1.5 │                                             │
│  │  - Price vs 20-Day Range│                                             │
│  │  - SMA 20/60 Alignment  │                                             │
│  │  - Choppiness Index     │                                             │
│  └────────────┬────────────┘                                             │
│               │ (Breakout Triggered)                                     │
│               ▼                                                          │
│  ┌─────────────────────────┐               ┌──────────────────────────┐  │
│  │  XGBoost Sentinel (ML)   │◄──────────────┤ Groq LLM Scorer          │  │
│  │  Calculates Win Prob     │◄──────────────┤ - 14 Chart Patterns      │  │
│  │  (Requires >= 75%)       │               │ - News Sentiment [-1, 1] │  │
│  └────────────┬────────────┘               └──────────────────────────┘  │
│               │                                                          │
│               ▼                                                          │
│  ┌─────────────────────────┐                                             │
│  │     Alerts Engine       │                                             │
│  │  - Theta-Decay Roll      │                                             │
│  │  - Telegram HTML Alert  │                                             │
│  └────────────┬────────────┘                                             │
│               │                                                          │
└───────────────┼──────────────────────────────────────────────────────────┘
                │
                ▼
      ┌──────────────────┐
      │  Telegram Alert  │
      └──────────────────┘
```

---

## ⚡ Trading Strategies & Option Rules

### 1. Daily & Intraday Setup Scanners
- **Trend Breakout / Breakdown**: Triggered when a stock breaks out above its 20-day high (or below its 20-day low) on strong volume expansion ($Z_{vol} \ge 1.5$), with moving averages aligned ($Price > SMA_{20} > SMA_{60}$ for Longs) and emerging momentum ($CHOP \le 55.0$).
- **Volume Breakout / Breakdown**: Enforces a higher institutional volume gate ($Z_{vol} \ge 2.0$) with similar breakout and momentum constraints.
- **Intraday Opening Range Breakout (ORB)**: Monitors the 15-minute Opening Range High/Low between 10:00 AM and 11:30 AM EST. Triggers on a high-volume crossing of the opening range boundaries, aligned with VWAP.

### 2. Chart Patterns Framework
The AI sentiment engine evaluates the 30-day normalized closing price series to identify classical price action structures before approving an alert:
- **Bullish Setups (Long)**: `Bull Flag`, `Cup with Handle`, `Ascending Triangle`, `Double Bottom`, `Inverted Head and Shoulders`, `Triple Bottom`, and `Gap-up Breakout` (vertical price gap with massive volume support establishing a rising staircase).
- **Bearish Setups (Short)**: `Bear Flag`, `Inverted Cup with Handle`, `Descending Triangle`, `Double Top`, `Head and Shoulders`, `Triple Top`, and `Gap-down Breakdown`.

### 3. Theta Decay Protection for Weekly Options
To safeguard against options theta decay, the engine implements a strict rolling expiration mechanism:
- **Swing / LEAPs**: Preferred for standard configurations to capture multi-week trends.
- **Weekly Options**: Fired only if conviction is exceptionally high:
  - Requires **Win Probability $\ge 80\%$**, extreme sentiment, and a `CONCORDANT` sentinel verdict.
  - **Dynamic Roll**: If the current Friday expiration is less than **3 days away (DTE < 3)**, the system automatically rolls the option target to the *following* week's Friday, ensuring theta does not degrade the trade value.

---

## 🧠 XGBoost Sentinel & Groq Sentiment Scorer

### 1. The Sentinel (XGBoost Meta-Classifier)
The sentinel acts as an AI confirmation gate. It evaluates quantitative inputs (RVOL, distance from SMA200, IV Rank, Bid-Ask spread, and GEX) to predict the statistical probability of a win. An alert is only released if `win_probability >= 75%` (or `80%` for Weekly timeframes).

### 2. Groq Llama-3.1-8B Scorer
- **Headlines & Price Action Scrutiny**: Scans the last 10 headlines for institutional catalysts (earnings, FDA decisions, major order book sweeps, insider buying) and checks for chart patterns.
- **Scale Normalization**: Correctly maps Groq's raw `[-1.00, 1.00]` sentiment score to the internal `[0.00, 1.00]` scale, where `0.50` represents neutral.
- **Divergence Guard (Hallucination Gate)**: If the absolute difference between the XGBoost quant prediction and the Groq sentiment score exceeds **0.25**, the alert is marked as a `BULLISH_HALLUCINATION` or `BEARISH_HALLUCINATION` and discarded.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.11
- Redis Server (Port `6379`)
- Tradier Account (Live or Sandbox API keys)
- Groq API Key

### 2. Installation
Clone the repository and install dependencies using python-venv or `uv` for speed:
```bash
# Set up virtual environment
python -m venv .venv311
.venv311\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Configuration
Configure the `.env` file located in `config/.env`:
```ini
GROQ_API_KEY=gsk_your_key_here
FINNHUB_TOKEN=your_token_here
TRADIER_ACCOUNT_ID=your_id
TRADIER_ACCESS_TOKEN=your_token
REDIS_URL=redis://localhost:6379/0
ADDITIONAL_DISCOVERY_TICKERS=AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD,NFLX,CRWV,UAL,SMCI,AVGO,QCOM,MU,NBIS,MRVL,RGTI,FCEL,JNJ,INTC,IREN
```

### 4. Running the Project
Use the included batch scripts for easy execution:
- **`start.bat`**: Runs the FastAPI app daemon and starts the background scanner.
- **`validate.bat`**: Runs the full CI/CD validation suite to verify scanner, sentinel, and health status.

---

## 🔍 Quant Research & Model Retraining Pipeline

RallyHunter features an offline, vectorized quant research pipeline located under the `research/` directory:

1. **`data_downloader.py`**: Downloads historical daily and intraday data for all watchlisted tickers (including `ADDITIONAL_DISCOVERY_TICKERS`) from Tradier and saves them as local pickles.
2. **`feature_engineer.py`**: Replays scanner breakout logic on historical data to build the XGBoost training set (`data/xgb_training_set.parquet`).
3. **`xgb_sentinel.py`**: Trains the machine learning model with Cross-Validation and generates metrics/weights.
4. **`optimizer.py`**: Optimizes the technical entry thresholds using historical simulations.
5. **`live_alert_auditor.py` / `data_auditor.py`**: Compares live and archived alerts against historical price paths to calculate mathematical parity and verify real-world win rates.
6. **Post-Market Automated Loop**: In production, the bot automatically wakes up every evening at **5:30 PM EST** to fetch the day's data, run feature engineering, retrain the XGBoost models, and dispatch a performance report to Telegram.

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Verifies server liveness and Redis socket connectivity. |
| `POST` | `/scan` | Triggers an on-demand manual scan for a list of tickers. |
| `GET` | `/watchlist` | Returns the active discovery watchlist. |
| `GET` | `/signals/{ticker}` | Retrieves the cached signal state for a specific stock. |
| `DELETE` | `/quiet-mode/{ticker}` | Manually clears the 4-hour cooldown quiet window. |

---

## 📁 Repository Directory Layout

```
Telegram Options bot/
├── config/
│   ├── .env                       ← Environment credentials & discovery tickers
│   └── config_optimized.json      ← Optimized strategy thresholds
├── data/
│   ├── historical/                ← Ticker pickle data (.pkl) for backtesting
│   ├── models/                    ← Saved XGBoost model binaries
│   └── rallyhunter.db             ← SQLite DB storing active & archived trade logs
├── research/
│   ├── data_downloader.py         ← Historical Tradier data downloader
│   ├── feature_engineer.py        ← Replays scanner logic & labels outcomes
│   ├── xgb_sentinel.py            ← Trains/validates the XGBoost meta-classifiers
│   ├── strategy_backtester.py     ← Vectorized offline backtester
│   ├── optimizer.py               ← Bayesian optimizer for strategy thresholds
│   ├── data_auditor.py            ← Audits parity of archived trade logs
│   └── live_alert_auditor.py      ← Audits live trade log performance
├── src/
│   ├── main.py                    ← FastAPI daemon, API routes & post-market loop
│   ├── scanner.py                 ├─ Core technical scanner & Groq AI scorer
│   ├── alerts.py                  ├─ Telegram client & cooldown supervisor
│   ├── indicators.py              ├─ Custom mathematical indicators (RSI, CHOP)
│   └── database.py                └─ SQLite CRUD operations
├── tests/
│   ├── health_check.py            ← Diagnostic script for configs, imports, & Redis
│   ├── test_strategies.py         ← Unit tests for weekly expiration & scanner logic
│   └── test_xgb_sentinel.py       ← Unit tests for XGBoost classifier
├── pyproject.toml                 ← Pytest & Ruff configuration rules
└── README.md                      ← This document
```
