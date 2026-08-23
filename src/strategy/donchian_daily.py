import os
import logging
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

log = logging.getLogger("rallyhunter.donchian")

class DonchianSwingStrategy:
    """
    Daily Donchian Channel Breakout/Breakdown Strategy.
    Implements the exact spec from Strategies.md:
      - R_20d / S_20d (20-day Donchian resistance/support)
      - Z_vol (volume z-score with projected volume)
      - CHOP_14 (Choppiness Index)
      - RSI_14
      - SMA_20 / SMA_60 trend filter
      - ATR-based exits
    
    Runs once per day after market close (or on a schedule) against daily bars.
    """

    def __init__(self, tradier_token: str = None):
        self.token = tradier_token or os.getenv("TRADIER_ACCESS_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }
        self.atr_cache = {}

    async def fetch_daily_bars(self, ticker: str, lookback_days: int = 80) -> pd.DataFrame:
        """Fetch daily OHLCV bars from Tradier."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        url = "https://api.tradier.com/v1/markets/history"
        params = {
            "symbol": ticker,
            "interval": "daily",
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d")
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params, timeout=10.0) as resp:
                    if resp.status != 200:
                        log.warning(f"Failed to fetch daily bars for {ticker}: {resp.status}")
                        return pd.DataFrame()

                    data = await resp.json()
                    history = data.get("history", {})
                    if not history or "day" not in history:
                        log.warning(f"No daily data for {ticker}")
                        return pd.DataFrame()

                    days = history["day"]
                    if not isinstance(days, list):
                        days = [days]

                    df = pd.DataFrame(days)
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df.dropna(subset=["close"])

        except Exception as e:
            log.error(f"Error fetching daily bars for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators required by the Donchian strategy."""
        if len(df) < 60:
            return pd.DataFrame()

        # Donchian 20-day channels (using prior 20 days, excluding today)
        df["R_20d"] = df["high"].shift(1).rolling(20).max()
        df["S_20d"] = df["low"].shift(1).rolling(20).min()

        # SMAs
        df["SMA_20"] = df["close"].rolling(20).mean()
        df["SMA_60"] = df["close"].rolling(60).mean()

        # RSI 14
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # ATR 14
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()

        # Choppiness Index 14
        atr_sum_14 = tr.rolling(14).sum()
        high_14 = df["high"].rolling(14).max()
        low_14 = df["low"].rolling(14).min()
        df["CHOP_14"] = 100 * np.log10(atr_sum_14 / (high_14 - low_14 + 1e-9)) / np.log10(14)

        # Volume Z-Score (20-day)
        vol_mean = df["volume"].shift(1).rolling(20).mean()
        vol_std = df["volume"].shift(1).rolling(20).std()
        df["Z_vol"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

        return df

    def evaluate_signals(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """Evaluate the latest bar for Donchian breakout/breakdown signals."""
        if df.empty or len(df) < 2:
            return []

        latest = df.iloc[-1]
        signals = []

        close = latest["close"]
        r20 = latest.get("R_20d")
        s20 = latest.get("S_20d")
        sma20 = latest.get("SMA_20")
        sma60 = latest.get("SMA_60")
        rsi = latest.get("RSI_14")
        chop = latest.get("CHOP_14")
        z_vol = latest.get("Z_vol")
        atr = latest.get("ATR_14")

        # Skip if any indicator is NaN
        if any(pd.isna(v) for v in [r20, s20, sma20, sma60, rsi, chop, z_vol, atr]):
            return []

        # ── LONG SETUPS ──
        if close > r20 and close > sma20 > sma60 and chop <= 55.0 and rsi <= 90:
            if z_vol >= 2.0:
                signals.append(self._build_signal(ticker, close, "Long", "Volume Breakout", z_vol, rsi, chop, atr, latest))
            elif z_vol >= 1.5:
                signals.append(self._build_signal(ticker, close, "Long", "Trend Breakout", z_vol, rsi, chop, atr, latest))

        # ── SHORT SETUPS ──
        if close < s20 and close < sma20 < sma60 and chop <= 55.0 and rsi >= 10:
            if z_vol >= 2.0:
                signals.append(self._build_signal(ticker, close, "Short", "Volume Breakdown", z_vol, rsi, chop, atr, latest))
            elif z_vol >= 1.5:
                signals.append(self._build_signal(ticker, close, "Short", "Trend Breakdown", z_vol, rsi, chop, atr, latest))

        return signals

    @staticmethod
    def _build_signal(ticker, close, direction, catalyst_type, z_vol, rsi, chop, atr, row) -> Dict:
        """Build a signal dict compatible with the SignalPipeline."""
        if direction == "Long":
            stop_loss = close - (1.5 * atr)
            take_profit = close + (2.0 * atr)
        else:
            stop_loss = close + (1.5 * atr)
            take_profit = close - (2.0 * atr)

        return {
            "ticker": ticker,
            "entry_price": close,
            "direction": direction,
            "catalyst_type": catalyst_type,
            "timestamp": datetime.now(),
            "timeframe": "SWING",
            "vwap_ratio": 1.0,  # Not applicable on daily bars
            "ema9_ratio": 1.0,
            "ema_trend_bullish": 1 if direction == "Long" else 0,
            "ema_trend_5m_bullish": 1 if direction == "Long" else 0,
            "hod_ratio": 1.0,
            "lod_ratio": 1.0,
            "volume": row.get("volume", 0),
            "z_vol": z_vol,
            "rsi_14": rsi,
            "chop_14": chop,
            "atr_14": atr,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "tf_confluence": f"Daily Donchian + SMA20/60 + CHOP({chop:.1f}) + RSI({rsi:.1f})"
        }

    async def scan_universe(self, tickers: List[str]) -> List[Dict]:
        """Scan all tickers for daily Donchian signals."""
        all_signals = []
        for ticker in tickers:
            try:
                df = await self.fetch_daily_bars(ticker)
                if df.empty:
                    continue
                df = self.compute_indicators(df)
                if df.empty:
                    continue
                self.atr_cache[ticker] = df["ATR_14"].iloc[-1]
                signals = self.evaluate_signals(df, ticker)
                if signals:
                    for sig in signals:
                        log.info(f"📅 SWING SIGNAL: {sig['catalyst_type']} on {ticker} at ${sig['entry_price']:.2f} (Z_vol={sig['z_vol']:.2f})")
                    all_signals.extend(signals)
            except Exception as e:
                log.error(f"Error scanning {ticker} for daily signals: {e}")
        return all_signals
