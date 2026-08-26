import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

def get_train_val_split(df: pd.DataFrame, features: list[str], target_col: str = "target"):
    """
    Splits a dataframe into train/val sets by date (80/20), falling 
    back to positional split if no 'date' column exists. Returns 
    (X_train, y_train, X_val, y_val).
    """
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        unique_dates = df["date"].dt.date.unique()
        split_date = unique_dates[int(len(unique_dates) * 0.8)]
        train_mask = df["date"].dt.date < split_date
        
        X_train = df[train_mask][features]
        y_train = df[train_mask][target_col]
        X_val = df[~train_mask][features]
        y_val = df[~train_mask][target_col]
    else:
        split_idx = int(len(df) * 0.8)
        X_train = df[features].iloc[:split_idx]
        y_train = df[target_col].iloc[:split_idx]
        X_val = df[features].iloc[split_idx:]
        y_val = df[target_col].iloc[split_idx:]
        
    return X_train, y_train, X_val, y_val

class XGBMicroSentinelV2:
    """
    XGBoost model tailored for identifying all-day trend potential (>5% intraday moves).
    This model evaluates intraday features (volume spikes, early ORB velocity).
    V2 Model runs on path-dependent labels.
    """
    def __init__(self):
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/models/xgb_micro_v2.json"))
        self._last_mtime = None
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.model = None
            self._is_trained = False
            self._load_model()
        except ImportError:
            self.xgb = None
            self.model = None
            log.error("XGBoost library is not installed.")
            
    @property
    def is_active(self) -> bool:
        return self._is_trained

    def _load_model(self):
        if self.xgb and os.path.exists(self.model_path):
            try:
                self.model = self.xgb.Booster()
                self.model.load_model(self.model_path)
                self._last_mtime = os.path.getmtime(self.model_path)
                self._is_trained = True
                log.info(f"Loaded trained XGBMicroSentinelV2 (mtime: {self._last_mtime}) from {self.model_path}")
            except Exception as e:
                log.warning(f"Failed to load V2 XGB model: {e}")

    def train(self, training_data_path: str):
        """Trains the model on the backtested labeled dataset."""
        if not self.xgb: return
        
        if not os.path.exists(training_data_path):
            log.error(f"Training data not found at {training_data_path}")
            return
            
        try:
            log.info(f"Training XGBMicroSentinelV2 on {training_data_path}...")
            df = pd.read_parquet(training_data_path)
            
            if len(df) < 10:
                log.warning("Not enough samples in training data to train effectively.")
                return
                
            features = [
                "relative_volume", "rsi_14", "chop_14", "expected_move_pct",
                "hist_vol_20", "sma20_ratio", "sma_spread", "breakout_pct", "direction_code"
            ]
            
            # Ensure all features exist
            for col in features:
                if col not in df.columns:
                    df[col] = 0.0

            df = df.dropna(subset=features + ["target"])
            
            X_train, y_train, X_val, y_val = get_train_val_split(df, features, "target")

            dtrain = self.xgb.DMatrix(X_train, label=y_train)
            dval = self.xgb.DMatrix(X_val, label=y_val)

            num_pos = np.sum(y_train == 1)
            num_neg = np.sum(y_train == 0)
            scale_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0

            params = {
                'max_depth': 4,
                'eta': 0.05,
                'objective': 'binary:logistic',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_weight,
                'eval_metric': 'logloss'
            }
            
            evals = [(dtrain, 'train'), (dval, 'val')]
            self.model = self.xgb.train(params, dtrain, num_boost_round=150, evals=evals, verbose_eval=False)
            self._is_trained = True
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_model(self.model_path)
            
            val_preds = self.model.predict(dval)
            val_labels = (val_preds > 0.5).astype(int)
            val_accuracy = np.mean(val_labels == y_val) if len(y_val) > 0 else 1.0
            
            log.info(f"XGBMicroSentinelV2 trained successfully on 14-21 DTE features! Validation Accuracy: {val_accuracy*100:.1f}%. Saved to {self.model_path}")
            
            return val_accuracy
        except Exception as e:
            log.error(f"XGBMicroSentinelV2 training failed: {e}")
            return 0.0

    def validate_setup(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if os.path.exists(self.model_path):
            current_mtime = os.path.getmtime(self.model_path)
            if self._last_mtime is None or current_mtime > self._last_mtime:
                log.info(f"🔄 HOT-RELOADING updated XGBoost model weights from disk into RAM (mtime: {current_mtime})...")
                self._load_model()

        if not self._is_trained or not self.model:
            return {"verdict": "NOT_CHECKED", "win_prob": 0.55}

        try:
            X_live = pd.DataFrame([{
                "relative_volume": float(features.get("relative_volume") or features.get("z_vol") or 1.5),
                "rsi_14": float(features.get("rsi_14", 55.0)),
                "chop_14": float(features.get("chop_14", 45.0)),
                "expected_move_pct": float(features.get("expected_move_pct", 4.0)),
                "hist_vol_20": float(features.get("hist_vol_20", 0.35)),
                "sma20_ratio": float(features.get("sma20_ratio", features.get("vwap_ratio", 1.0))),
                "sma_spread": float(features.get("sma_spread", 0.02)),
                "breakout_pct": float(features.get("breakout_pct", 0.01)),
                "direction_code": int(features.get("direction_code", 1))
            }])
            if hasattr(self.model, "feature_names") and self.model.feature_names:
                cols = [c for c in self.model.feature_names if c in X_live.columns]
                X_live = X_live[cols]
                
            dtest = self.xgb.DMatrix(X_live)
            prob = self.model.predict(dtest)[0]
            
            verdict = "CONCORDANT" if prob >= 0.75 else "HALLUCINATION"
            return {"verdict": verdict, "win_prob": float(prob)}
        except Exception as e:
            log.warning(f"XGBMicroSentinelV2 validation error: {e}")
            return {"verdict": "ERROR", "win_prob": 0.50}
