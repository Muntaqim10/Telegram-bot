import os
import glob
import asyncio
import logging
import pandas as pd
from typing import List

log = logging.getLogger(__name__)

class IntradayBacktester:
    """
    Simulates the live event-driven engine using historical 1-minute data to 
    identify setups that resulted in successful >5% all-day trends.
    Generates a labeled dataset for XGBoost training.
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
        log.info("Loading historical intraday data...")
        historical_data = self.load_historical_data(data_dir)
        
        for ticker, df in historical_data.items():
            log.info(f"Backtesting {ticker}...")
            
            # Group data by trading day
            days = df.groupby(df.index.date)
            
            for date, daily_df in days:
                self.engine.orb_strategy.reset_daily()
                signal_fired = False
                signal_entry_price = 0.0
                signal_direction = ""
                
                # We need features at the exact moment of the breakout
                features_at_entry = {}
                
                stop_loss = 0.0
                take_profit = 0.0
                
                for timestamp, row in daily_df.iterrows():
                    price = row["close"]
                    volume = row["volume"]
                    
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
                            
                            # Establish SL/TP
                            current_atr = 1.5
                            if ticker in self.engine.orb_strategy.intraday_state and "atr" in self.engine.orb_strategy.intraday_state[ticker]:
                                current_atr = self.engine.orb_strategy.intraday_state[ticker]["atr"]
                            
                            atr_dist = current_atr * self.engine.risk_manager.atr_multiplier
                            if signal_direction == "Long":
                                stop_loss = signal_entry_price - atr_dist
                                take_profit = signal_entry_price + (atr_dist * 2.0)
                            else:
                                stop_loss = signal_entry_price + atr_dist
                                take_profit = signal_entry_price - (atr_dist * 2.0)
                            
                            features_at_entry = {
                                "ticker": ticker,
                                "date": date,
                                "direction": signal_direction,
                                "entry_price": signal_entry_price,
                                "entry_time_minute": timestamp.hour * 60 + timestamp.minute,
                                "entry_volume": volume,
                                "vwap_ratio": signal.get("vwap_ratio", 1.0),
                                "ema9_ratio": signal.get("ema9_ratio", 1.0),
                                "ema_trend": signal.get("ema_trend_bullish", 1),
                                "ema_trend_5m": signal.get("ema_trend_5m_bullish", 1),
                                "spy_correlation": spy_corr,
                                "hod_ratio": signal.get("hod_ratio", 1.0),
                                "lod_ratio": signal.get("lod_ratio", 1.0)
                            }
                    else:
                        # Path-aware evaluation
                        if signal_direction == "Long":
                            if price <= stop_loss:
                                features_at_entry["target"] = 0
                                features_at_entry["max_gain"] = (price - signal_entry_price) / signal_entry_price
                                self.training_data.append(features_at_entry)
                                break
                            elif price >= take_profit:
                                features_at_entry["target"] = 1
                                features_at_entry["max_gain"] = (price - signal_entry_price) / signal_entry_price
                                self.training_data.append(features_at_entry)
                                break
                        else: # Short
                            if price >= stop_loss:
                                features_at_entry["target"] = 0
                                features_at_entry["max_gain"] = (signal_entry_price - price) / signal_entry_price
                                self.training_data.append(features_at_entry)
                                break
                            elif price <= take_profit:
                                features_at_entry["target"] = 1
                                features_at_entry["max_gain"] = (signal_entry_price - price) / signal_entry_price
                                self.training_data.append(features_at_entry)
                                break
                else:
                    # End of Day Evaluation if SL/TP never hit
                    if signal_fired and "target" not in features_at_entry:
                        features_at_entry["target"] = 0 # Default to 0 if TP not hit
                        if signal_direction == "Long":
                            features_at_entry["max_gain"] = (price - signal_entry_price) / signal_entry_price
                        else:
                            features_at_entry["max_gain"] = (signal_entry_price - price) / signal_entry_price
                        self.training_data.append(features_at_entry)
                    
        log.info(f"Backtest complete. Generated {len(self.training_data)} training samples.")
        
        # Save to parquet for ML training
        if self.training_data:
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "ml_training_data.parquet")
            
            tdf = pd.DataFrame(self.training_data)
            tdf.to_parquet(out_path)
            log.info(f"Training dataset saved to {out_path} (Success Rate: {tdf['target'].mean()*100:.1f}%)")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.main import IntradayEngine
    import redis.asyncio as aioredis
    
    logging.basicConfig(level=logging.INFO)
    
    async def run():
        redis = await aioredis.from_url("redis://localhost", decode_responses=True)
        engine = IntradayEngine(redis)
        backtester = IntradayBacktester(engine)
        
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/intraday/"))
        await backtester.run_simulation(data_dir)
        await redis.aclose()
        
    asyncio.run(run())
