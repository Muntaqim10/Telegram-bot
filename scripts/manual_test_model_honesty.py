"""Manual test: the bot must not display confidence it does not have.

Two independent ways the conviction score becomes meaningless:
  1. the strategy that produced the signal never computed the model's features, so the
     model is scoring hardcoded defaults -- the same number on every alert of that type
  2. the model itself cannot separate inputs, so its output is constant

Either one must show as UNSCORED rather than a tier and a percentage.

Run: python scripts/manual_test_model_honesty.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.xgb_micro_v2 import MIN_USEFUL_SPREAD, PROBE_GRID, XGBMicroSentinelV2
from src.models.signal import TradeSignal
from src.utils.telegram_formatter import format_telegram_alert

failures = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


print("\n1. The discrimination probe measures the loaded model")
m = XGBMicroSentinelV2()
if not m.is_active:
    print("  (no trained model on disk; skipping model-side checks)")
else:
    check("probe ran and recorded a spread", m.prediction_spread > 0.0,
          f"spread {m.prediction_spread:.6f}")
    check("verdict reflects the probe result",
          m.is_discriminating == (m.prediction_spread >= MIN_USEFUL_SPREAD))
    grid_points = 1
    for v in PROBE_GRID.values():
        grid_points *= len(v)
    check("probe sweeps a real grid, not one point", grid_points >= 100, f"{grid_points} points")

print("\n2. A model that cannot discriminate reports NO_DISCRIMINATION")
if m.is_active:
    real_flag, real_spread = m.is_discriminating, m.prediction_spread
    m.is_discriminating, m.prediction_spread = False, 0.0001
    r = m.validate_setup({"sma_spread": 0.08, "sma20_ratio": 1.3,
                          "rsi_14": 77.0, "direction_code": 1})
    check("verdict is NO_DISCRIMINATION", r["verdict"] == "NO_DISCRIMINATION", f"got {r['verdict']}")
    check("spread is reported alongside", "prediction_spread" in r)
    m.is_discriminating, m.prediction_spread = real_flag, real_spread
    r = m.validate_setup({"sma_spread": 0.08, "sma20_ratio": 1.3,
                          "rsi_14": 77.0, "direction_code": 1})
    check("a discriminating model still returns a real verdict",
          r["verdict"] in ("CONCORDANT", "AMBIGUOUS", "HALLUCINATION"), f"got {r['verdict']}")

print("\n3. An UNSCORED alert shows no percentage")
base = dict(
    ticker="PLTR", price=150.0, signal_direction="Long", strategy_type="ORB Breakout",
    timestamp="2026-09-03 10:00:00", win_probability=0.3654, xgb_win_prob=0.3654,
    sentinel_verdict="UNSCORED_NO_FEATURES", context_score="Intraday ORB",
    catalyst="Test", historical_edge="n/a", stop_loss=145.0, take_profit=160.0,
)
unscored = format_telegram_alert(TradeSignal(conviction="⚪ UNSCORED", **base))
check("says UNSCORED", "UNSCORED" in unscored)
check("explains why", "not discriminating" in unscored or "no information" in unscored)
check("does NOT print a confidence percentage", "36%" not in unscored and "37%" not in unscored,
      "a percentage leaked into an unscored alert")

print("\n4. A properly scored alert still shows its number")
scored = format_telegram_alert(TradeSignal(conviction="🟢 HIGH", **{**base, "win_probability": 0.42}))
check("tier shown", "HIGH" in scored)
check("percentage shown", "42%" in scored, "expected 42% in the scored alert")

print("\n5. Feature completeness is what the pipeline keys on")
import inspect

from src.execution.signal_pipeline import SignalPipeline
src = inspect.getsource(SignalPipeline.process_signal)
check("pipeline checks for missing features", "missing = [" in src)
check("pipeline no longer scores hardcoded defaults",
      'signal.get("sma_spread", 0.02)' not in src,
      "the 0.02/55.0 fallbacks are what produced two scores for every alert")
check("conviction requires supplied features", "features_supplied and" in src)
check("the win_prob gate is skipped when unscored", "model_scores and (win_prob" in src)

print("\n" + "=" * 62)
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
