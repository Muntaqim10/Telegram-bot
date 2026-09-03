import os
import json
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

MODEL_FEATURES = [
    "sma_spread", "sma20_ratio", "rsi_14", "direction_code"
]

# Feature ranges observed in the training data, used to probe whether the loaded model
# actually responds to its inputs. A model trained on features with no signal collapses
# to predicting the base rate for everything -- honest, but it means the conviction
# tiers and win_prob gates downstream are ranking nothing.
PROBE_GRID = {
    "sma_spread": [-0.35, -0.05, 0.0, 0.08, 1.30],
    "sma20_ratio": [0.25, 0.90, 1.04, 1.30, 3.10],
    "rsi_14": [10.0, 30.0, 60.0, 77.0, 89.0],
    "direction_code": [0, 1],
}
# Below this spread across the whole grid, the score carries no information.
MIN_USEFUL_SPREAD = 0.02

class XGBMicroSentinelV2:
    """
    XGBoost model tailored for identifying high-probability options continuation.
    Enforces monotonic constraints and shallow trees to eliminate overfitting.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/models/xgb_micro_v2.json")
        )
        self.candidate_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/models/xgb_micro_v2_candidate.json")
        )
        self.model = None
        self._is_trained = False
        self._last_mtime = None
        # Set by _probe_discrimination() every time weights are loaded.
        self.is_discriminating = False
        self.prediction_spread = 0.0
        
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            log.warning("xgboost not installed. XGBMicroSentinelV2 disabled.")
            self.xgb = None
            
        self._load_model()

    @property
    def is_active(self) -> bool:
        return self._is_trained

    def _load_model(self):
        if not self.xgb: return
        if os.path.exists(self.model_path):
            try:
                self.model = self.xgb.Booster()
                self.model.load_model(self.model_path)
                self._is_trained = True
                self._last_mtime = os.path.getmtime(self.model_path)
                log.info(f"Loaded trained XGBMicroSentinelV2 (mtime: {self._last_mtime}) from {self.model_path}")
                self._probe_discrimination()
            except Exception as e:
                log.error(f"Failed to load XGBoost model from {self.model_path}: {e}")
                self._is_trained = False
        else:
            log.warning(f"No trained model found at {self.model_path}")

    def train(self, training_data_path: str):
        """Trains the model using regularized shallow trees with monotonic constraints."""
        if not self.xgb: return 0.0
        
        if not os.path.exists(training_data_path):
            log.error(f"Training data not found at {training_data_path}")
            return 0.0
            
        try:
            log.info(f"Training regularized XGBMicroSentinelV2 on {training_data_path}...")
            df = pd.read_parquet(training_data_path)
            
            if len(df) < 10:
                log.warning("Not enough samples in training data to train effectively.")
                return 0.0
                
            features = list(MODEL_FEATURES)
            
            # Ensure all features exist
            for col in features:
                if col not in df.columns:
                    df[col] = 0.0

            df = df.dropna(subset=features + ["target"])
            
            X_train, y_train, X_val, y_val = get_train_val_split(df, features, "target")

            dtrain = self.xgb.DMatrix(X_train, label=y_train)
            dval = self.xgb.DMatrix(X_val, label=y_val)

            # Strict anti-overfitting & calibration parameters
            params = {
                'max_depth': 2,
                'eta': 0.03,
                'objective': 'binary:logistic',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1.0,
                'reg_alpha': 2.0,
                'reg_lambda': 10.0,
                'monotone_constraints': '(1, 1, 1, 0)',
                'eval_metric': 'logloss'
            }
            
            evals = [(dtrain, 'train'), (dval, 'val')]
            self.model = self.xgb.train(params, dtrain, num_boost_round=35, evals=evals, verbose_eval=False)
            self._is_trained = True
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_model(self.model_path)
            self.model.save_model(self.candidate_path)
            
            # Save feature columns definition
            feat_cols_path = os.path.join(os.path.dirname(self.model_path), "feature_columns.json")
            with open(feat_cols_path, "w", encoding="utf-8") as f:
                json.dump(features, f, indent=2)
            
            val_preds = self.model.predict(dval)
            val_labels = (val_preds > 0.36).astype(int)
            val_accuracy = float(np.mean(val_labels == y_val)) if len(y_val) > 0 else 1.0
            
            log.info(f"XGBMicroSentinelV2 trained successfully! Validation Accuracy: {val_accuracy*100:.1f}%. Saved to {self.model_path} and {self.candidate_path}")
            
            return val_accuracy
        except Exception as e:
            log.error(f"XGBMicroSentinelV2 training failed: {e}")
            return 0.0

    def _probe_discrimination(self) -> None:
        """Checks whether the loaded model's score actually varies with its inputs.

        Sweeps each feature across the range seen in training and measures the spread of
        the resulting predictions. A model fitted on features with no signal correctly
        collapses to the base rate -- but then the conviction tiers and the win_prob gates
        are sorting noise, and an alert that prints "HIGH conviction" is claiming
        information nobody has. Re-run on every hot reload, so a retrain that restores
        discrimination turns the tiers back on by itself.
        """
        self.is_discriminating = False
        self.prediction_spread = 0.0
        if not self._is_trained or not self.model:
            return
        try:
            import itertools
            rows = []
            keys = list(PROBE_GRID)
            for combo in itertools.product(*(PROBE_GRID[k] for k in keys)):
                rows.append(dict(zip(keys, combo)))
            X = pd.DataFrame(rows)
            if hasattr(self.model, "feature_names") and self.model.feature_names:
                cols = [c for c in self.model.feature_names if c in X.columns]
                X = X[cols]
            preds = self.model.predict(self.xgb.DMatrix(X))
            self.prediction_spread = float(preds.max() - preds.min())
            self.is_discriminating = self.prediction_spread >= MIN_USEFUL_SPREAD

            if self.is_discriminating:
                log.info(f"XGB discrimination check: spread {self.prediction_spread:.4f} "
                         f"across {len(rows)} probe points. Conviction tiers active.")
            else:
                log.error(
                    f"XGB MODEL NOT DISCRIMINATING: predictions span only "
                    f"{self.prediction_spread:.6f} across {len(rows)} probe points "
                    f"(range {preds.min():.4f}-{preds.max():.4f}). The score cannot rank "
                    f"setups, so conviction tiers and the win_prob gate are DISABLED. "
                    f"Retrain on features with signal -- see scripts/check_feature_signal.py."
                )
        except Exception as e:
            log.warning(f"XGB discrimination probe failed: {e}")

    def validate_setup(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if os.path.exists(self.model_path):
            current_mtime = os.path.getmtime(self.model_path)
            if self._last_mtime is None or current_mtime > self._last_mtime:
                log.info(f"🔄 HOT-RELOADING updated XGBoost model weights from disk into RAM (mtime: {current_mtime})...")
                self._load_model()

        if not self._is_trained or not self.model:
            return {"verdict": "NOT_CHECKED", "win_prob": 0.50}

        try:
            X_live = pd.DataFrame([{
                "sma_spread": float(features.get("sma_spread", 0.02)),
                "sma20_ratio": float(features.get("sma20_ratio", features.get("vwap_ratio", 1.0))),
                "rsi_14": float(features.get("rsi_14", 55.0)),
                "direction_code": int(features.get("direction_code", 1))
            }])
            if hasattr(self.model, "feature_names") and self.model.feature_names:
                cols = [c for c in self.model.feature_names if c in X_live.columns]
                X_live = X_live[cols]
                
            dtest = self.xgb.DMatrix(X_live)
            prob = self.model.predict(dtest)[0]
            
            # A constant score cannot separate setups; say so instead of dressing it up
            # as a verdict the downstream gates will act on.
            if not self.is_discriminating:
                return {"verdict": "NO_DISCRIMINATION", "win_prob": float(prob),
                        "prediction_spread": self.prediction_spread}

            # Calibrated probability thresholds (42.1% win rate on High, 18.0% on Low)
            if prob >= 0.375:
                verdict = "CONCORDANT"
            elif prob >= 0.360:
                verdict = "AMBIGUOUS"
            else:
                verdict = "HALLUCINATION"
            return {"verdict": verdict, "win_prob": float(prob)}
        except Exception as e:
            log.warning(f"XGBMicroSentinelV2 validation error: {e}")
            return {"verdict": "ERROR", "win_prob": 0.50}
