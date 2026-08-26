import logging
import os
import time
import aiohttp
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

class TimeframeConfluenceEngine:
    """
    Evaluates multi-timeframe confluence specifically aligned to the option expiration cycle:
    
    1. WEEKLY TRADING CYCLE (5-14 DTE):
       - Macro Trend (Weekly Chart): Multi-week support/resistance & 10-week EMA regime
       - Confirmation (Daily Chart): Daily 20 EMA & Donchian momentum alignment
       - Intraday Trend (VWAP): Institutional VWAP hold
       
    2. 3-DAY SHORT CYCLE (1-3 DTE):
       - Primary Trend (Daily Chart): Multi-session trend & 20 EMA structure
       - Intraday Trend (VWAP): Real-time institutional VWAP hold / pressure
       - Momentum Trigger (EMA9): Fine-tuned 9 EMA momentum push / rejection
    """
    def __init__(self, tradier_token: Optional[str] = None):
        self.token = tradier_token or os.getenv("TRADIER_ACCESS_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }
        # In-memory bar cache (5-minute TTL) to guarantee 0 redundant network calls
        self._history_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

    async def fetch_historical_bars(self, ticker: str, interval: str = "daily", days: int = 250, session: Optional[aiohttp.ClientSession] = None) -> pd.DataFrame:
        """Fetch historical bars with in-memory caching."""
        cache_key = f"{ticker}_{interval}_{days}"
        now_ts = time.time()
        
        if cache_key in self._history_cache:
            cached_time, df = self._history_cache[cache_key]
            if now_ts - cached_time < 300.0:  # 5-minute cache
                return df

        if not self.token:
            return pd.DataFrame()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        url = "https://api.tradier.com/v1/markets/history"
        params = {
            "symbol": ticker,
            "interval": interval,
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d")
        }

        try:
            async def do_get(s):
                async with s.get(url, headers=self.headers, params=params, timeout=10.0) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return None

            if session:
                data = await do_get(session)
            else:
                async with aiohttp.ClientSession() as s:
                    data = await do_get(s)

            if not data:
                return pd.DataFrame()

            days_data = data.get("history", {}).get("day", [])
            if not isinstance(days_data, list):
                days_data = [days_data]

            df = pd.DataFrame(days_data)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                self._history_cache[cache_key] = (now_ts, df)
                return df
        except Exception as e:
            log.warning(f"Failed to fetch historical {interval} bars for {ticker}: {e}")

        return pd.DataFrame()

    async def evaluate_confluence(
        self,
        ticker: str,
        spot_price: float,
        direction: str,
        dte: int,
        intraday_state: Optional[Dict[str, Any]] = None,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the full 3-timeframe triad based on expiration DTE:
        - If DTE <= 3: Evaluates Daily -> Intraday VWAP -> EMA9 Momentum (3-Day Short Cycle)
        - If DTE > 3:  Evaluates Weekly -> Daily -> Intraday VWAP (Weekly Swing Cycle)
        """
        is_3day_cycle = (dte <= 3)
        cycle_name = "3-DAY SHORT CYCLE" if is_3day_cycle else "WEEKLY SWING CYCLE"
        is_long = (direction.lower() == "long")

        # Pull Daily Historical Bars
        daily_df = await self.fetch_historical_bars(ticker, interval="daily", days=250, session=session)
        data_quality_warning = False

        if daily_df is None or daily_df.empty or len(daily_df) < 30:
            log.warning(f"[{ticker}] Timeframe confluence: Insufficient daily bar history ({len(daily_df) if daily_df is not None else 0} bars). Bypassing gate with data quality warning.")
            return {
                "cycle_name": cycle_name,
                "is_3day_cycle": is_3day_cycle,
                "concordant": True,  # Fail open intentionally so external data outage does not block all alerts
                "confluence_summary": "INSUFFICIENT_DATA",
                "data_quality_warning": True,
                "tf_1": {"name": "Daily (Primary Trend)", "status": True, "desc": "⚠️ Insufficient historical data (<30 bars)"},
                "tf_2": {"name": "Intraday Trend (VWAP)", "status": True, "desc": "⚠️ Data check bypassed"},
                "tf_3": {"name": "Momentum Trigger (EMA9)", "status": True, "desc": "⚠️ Data check bypassed"},
            }

        # 1. Evaluate Daily Metric
        daily_bullish = True
        daily_desc = "Neutral / Breakout Pending"
        
        if len(daily_df) >= 20:
            ema20 = daily_df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
            high_20d = daily_df["high"].iloc[-21:-1].max() if len(daily_df) >= 21 else daily_df["high"].max()
            low_20d = daily_df["low"].iloc[-21:-1].min() if len(daily_df) >= 21 else daily_df["low"].min()

            if is_long:
                daily_bullish = (spot_price >= ema20) or (spot_price >= high_20d * 0.98)
                daily_desc = f"Above 20 EMA (${ema20:.2f})" if spot_price >= ema20 else f"Near 20D High (${high_20d:.2f})"
            else:
                daily_bullish = (spot_price <= ema20) or (spot_price <= low_20d * 1.02)
                daily_desc = f"Below 20 EMA (${ema20:.2f})" if spot_price <= ema20 else f"Near 20D Low (${low_20d:.2f})"
        else:
            data_quality_warning = True
            daily_desc = "⚠️ Daily EMA history limited"

        # 2. Evaluate Weekly Metric (for Weekly Cycle)
        weekly_bullish = True
        weekly_desc = "Multi-Week Macro Aligned"
        
        if len(daily_df) >= 50:
            weekly_df = daily_df.resample('W-FRI').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            
            if len(weekly_df) >= 10:
                w_ema10 = weekly_df["close"].ewm(span=10, adjust=False).mean().iloc[-1]
                if is_long:
                    weekly_bullish = spot_price >= w_ema10
                    weekly_desc = f"Macro Bullish > 10W EMA (${w_ema10:.2f})"
                else:
                    weekly_bullish = spot_price <= w_ema10
                    weekly_desc = f"Macro Bearish < 10W EMA (${w_ema10:.2f})"
            else:
                data_quality_warning = True
                weekly_desc = "⚠️ Weekly history limited (<10 weeks)"
        else:
            data_quality_warning = True
            weekly_desc = "⚠️ Insufficient bars for weekly resample (<50 days)"

        # 3. Evaluate Intraday VWAP Hold Metric
        vwap_bullish = True
        vwap_desc = "VWAP Hold Concordant"
        
        if intraday_state:
            vwap = intraday_state.get("vwap", spot_price)
            lod = intraday_state.get("lod", spot_price * 0.98)
            hod = intraday_state.get("hod", spot_price * 1.02)
            
            if is_long:
                vwap_bullish = spot_price >= vwap or spot_price >= (hod * 0.99)
                vwap_desc = f"Hold > VWAP (${vwap:.2f})"
            else:
                vwap_bullish = spot_price <= vwap or spot_price <= (lod * 1.01)
                vwap_desc = f"Pressure < VWAP (${vwap:.2f})"
        else:
            data_quality_warning = True
            vwap_desc = "⚠️ Intraday state missing"

        # 4. Evaluate Intraday EMA9 Momentum Trigger Metric (for 3-Day Cycle)
        ema9_bullish = True
        ema9_desc = "EMA9 Momentum Push Trigger"
        
        if intraday_state:
            ema9 = intraday_state.get("ema9", spot_price)
            if is_long:
                ema9_bullish = spot_price >= ema9
                ema9_desc = f"Push > 9 EMA (${ema9:.2f})"
            else:
                ema9_bullish = spot_price <= ema9
                ema9_desc = f"Rejection < 9 EMA (${ema9:.2f})"
        else:
            data_quality_warning = True
            ema9_desc = "⚠️ Intraday state missing"

        # Assemble Timeframe Matrix based on Cycle
        if is_3day_cycle:
            # 3-Day Cycle: Daily (Trend) -> Intraday (VWAP) -> Momentum (EMA9)
            tf_1_name = "Daily (Primary Trend)"
            tf_1_status = daily_bullish
            tf_1_desc = daily_desc

            tf_2_name = "Intraday Trend (VWAP)"
            tf_2_status = vwap_bullish
            tf_2_desc = vwap_desc

            tf_3_name = "Momentum Trigger (EMA9)"
            tf_3_status = ema9_bullish
            tf_3_desc = ema9_desc

            all_concordant = (tf_1_status and tf_2_status and tf_3_status)
            confluence_str = "Daily (Trend) -> Intraday VWAP -> EMA9 Momentum"
        else:
            # Weekly Swing Cycle: Weekly (Macro) -> Daily (Confirmation) -> Intraday (VWAP)
            tf_1_name = "Weekly (Macro Trend)"
            tf_1_status = weekly_bullish
            tf_1_desc = weekly_desc

            tf_2_name = "Daily (Confirmation)"
            tf_2_status = daily_bullish
            tf_2_desc = daily_desc

            tf_3_name = "Intraday Trend (VWAP)"
            tf_3_status = vwap_bullish
            tf_3_desc = vwap_desc

            all_concordant = (tf_1_status and tf_2_status and tf_3_status)
            confluence_str = "Weekly (Macro) -> Daily (Confirmation) -> Intraday VWAP"

        matrix = {
            "cycle_name": cycle_name,
            "is_3day_cycle": is_3day_cycle,
            "concordant": all_concordant,
            "confluence_summary": confluence_str,
            "data_quality_warning": data_quality_warning,
            "tf_1": {"name": tf_1_name, "status": tf_1_status, "desc": tf_1_desc},
            "tf_2": {"name": tf_2_name, "status": tf_2_status, "desc": tf_2_desc},
            "tf_3": {"name": tf_3_name, "status": tf_3_status, "desc": tf_3_desc},
        }

        return matrix
