import os
import glob
import asyncio
import logging
import pandas as pd
from src.utils.math_utils import calculate_take_profit

log = logging.getLogger(__name__)

class IntradayBacktesterV2:
    """
    Simulates the live event-driven engine using historical 1-minute data to 
    identify setups that resulted in successful trends.
    Uses path-dependent 5-day evaluation with daily ATR calculations.
    """
    def __init__(self, engine):
        self.engine = engine # Instance of IntradayEngine
        self.training_data = []
        
    def load_historical_data(self, data_dir: str) -> dict:
        """Loads downloaded 1-minute CSVs for the ML training universe."""
        data = {}
        csv_files = glob.glob(os.path.join(data_dir, "*_1m.csv"))
        for file in csv_files:
            ticker = os.path.basename(file).split("_")[0]
            df = pd.read_csv(file, parse_dates=["time"], index_col="time")
            data[ticker] = df
        return data

    async def run_simulation(self, data_dir: str):
        """
        Feeds historical data into the live strategy engine.
        Tracks the success of breakouts to generate a training set.
        """
        from src.strategy.donchian_daily import DonchianSwingStrategy
        
        log.info("Loading historical intraday data for V2 path-dependent testing...")
        historical_data = self.load_historical_data(data_dir)
        donchian = DonchianSwingStrategy("dummy_token")
        
        for ticker, df in historical_data.items():
            log.info(f"Backtesting {ticker} (V2)...")
            
            # Resample 1m data to daily to compute ATR_14
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            if 'volume' in df.columns:
                agg_dict['volume'] = 'sum'
                
            daily_df = df.resample('D').agg(agg_dict).dropna()
            
            # Compute indicators (this will add ATR_14)
            daily_df = donchian.compute_indicators(daily_df)
            
            # Group data by trading day
            days = df.groupby(df.index.date)
            
            for date, daily_tick_df in days:
                self.engine.orb_strategy.reset_daily()
                signal_fired = False
                
                features_at_entry = {}
                
                for timestamp, row in daily_tick_df.iterrows():
                    price = row["close"]
                    volume = row.get("volume", 0)
                    
                    # Concurrently feed SPY tick to maintain Mega-Cap state
                    spy_df = historical_data.get("SPY")
                    if ticker != "SPY" and spy_df is not None:
                        try:
                            spy_row = spy_df.loc[timestamp]
                            if isinstance(spy_row, pd.DataFrame): spy_row = spy_row.iloc[0]
                            self.engine.orb_strategy.process_tick(
                                "SPY", spy_row["close"], spy_row["volume"], timestamp
                            )
                        except KeyError:
                            pass # SPY missing this exact minute
                    
                    # 1. Process Tick
                    if not signal_fired:
                        signal = self.engine.orb_strategy.process_tick(
                            ticker=ticker, price=price, volume=volume, timestamp=timestamp
                        )
                        
                        if signal:
                            signal_fired = True
                            signal_entry_price = signal["entry_price"]
                            signal_direction = signal["direction"]
                            
                            # Capture simple features at entry
                            spy_state = self.engine.orb_strategy.intraday_state.get("SPY", {})
                            spy_vwap = spy_state.get("vwap")
                            spy_corr = spy_state.get("last_price") / spy_vwap if spy_vwap else 1.0
                            
                            features_at_entry = {
                                "ticker": ticker,
                                "date": date,
                                "direction": signal_direction,
                                "entry_price": signal_entry_price,
                                "entry_time_minute": timestamp.hour * 60 + timestamp.minute,
                                "relative_volume": signal.get("relative_volume", 1.0),
                                "vwap_ratio": signal.get("vwap_ratio", 1.0),
                                "ema9_ratio": signal.get("ema9_ratio", 1.0),
                                "ema_trend": signal.get("ema_trend_bullish", 1),
                                "ema_trend_5m": signal.get("ema_trend_5m_bullish", 1),
                                "spy_correlation": spy_corr,
                                "hod_ratio": signal.get("hod_ratio", 1.0),
                                "lod_ratio": signal.get("lod_ratio", 1.0)
                            }
                            
                            # Lookup previous day's ATR (to prevent lookahead bias)
                            past_daily = daily_df.loc[:date]
                            if len(past_daily) > 1:
                                prev_day = past_daily.iloc[-2]
                                current_atr = prev_day.get("ATR_14", 1.5)
                            else:
                                current_atr = 1.5
                            
                            if pd.isna(current_atr) or current_atr <= 0:
                                current_atr = 1.5
                                
                            atr_dist = current_atr * self.engine.risk_manager.atr_multiplier
                            if signal_direction == "Long":
                                stop_loss = signal_entry_price - atr_dist
                                take_profit = calculate_take_profit(signal_entry_price, current_atr, signal_direction)
                            else:
                                stop_loss = signal_entry_price + atr_dist
                                take_profit = calculate_take_profit(signal_entry_price, current_atr, signal_direction)
                                
                            # Path Evaluation over 5 forward trading days
                            future_df = df.loc[timestamp:]
                            unique_days = pd.Series(future_df.index.date).unique()
                            if len(unique_days) > 5:
                                end_date = unique_days[5]
                                future_df = future_df.loc[:str(end_date)]
                                
                            target = 0
                            max_gain = 0.0
                            for future_ts, future_row in future_df.iterrows():
                                future_price = future_row["close"]
                                
                                if signal_direction == "Long":
                                    gain = (future_price - signal_entry_price) / signal_entry_price
                                    if gain > max_gain: max_gain = gain
                                    if future_price <= stop_loss:
                                        target = 0
                                        break
                                    elif future_price >= take_profit:
                                        target = 1
                                        break
                                else:
                                    gain = (signal_entry_price - future_price) / signal_entry_price
                                    if gain > max_gain: max_gain = gain
                                    if future_price >= stop_loss:
                                        target = 0
                                        break
                                    elif future_price <= take_profit:
                                        target = 1
                                        break
                                        
                            features_at_entry["target"] = target
                            features_at_entry["max_gain"] = max_gain
                            self.training_data.append(features_at_entry)
                            break # Move to next day (only evaluate the first signal per day)
                            
        log.info(f"Backtest V2 complete. Generated {len(self.training_data)} training samples.")
        
        # Save to parquet for ML training
        if self.training_data:
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "ml_training_data_v2.parquet")
            
            tdf = pd.DataFrame(self.training_data)
            tdf.to_parquet(out_path)
            log.info(f"V2 Training dataset saved to {out_path} (Success Rate: {tdf['target'].mean()*100:.1f}%)")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.main import IntradayEngine
    import redis.asyncio as aioredis
    
    logging.basicConfig(level=logging.INFO)
    
    async def run():
        redis = await aioredis.from_url("redis://localhost", decode_responses=True)
        engine = IntradayEngine(redis)
        backtester = IntradayBacktesterV2(engine)
        
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/intraday/"))
        await backtester.run_simulation(data_dir)
        await redis.aclose()
        
    asyncio.run(run())
