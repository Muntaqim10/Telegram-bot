"""Manual test for the configurable option DTE window.

The bot used to hardcode a 14-21 day target. This account's own results are worst in
that bucket (42.1% win, -$2,757 net across 57 contracts) and best at 0-1 DTE, so the
window has to be a setting rather than a constant -- and every place that echoes it,
including alert headers, has to follow the setting rather than a stale string.

Run: python scripts/manual_test_dte_config.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

failures = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def reload_with(**env):
    """Re-import the pricer under a given environment."""
    for k in ("OPTION_MIN_DTE", "OPTION_MAX_DTE"):
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    import src.execution.options_pricer as op
    importlib.reload(op)
    return op


print("\n1. Defaults preserve the existing behaviour")
op = reload_with()
check("min defaults to 14", op.TARGET_MIN_DTE == 14, f"got {op.TARGET_MIN_DTE}")
check("max defaults to 21", op.TARGET_MAX_DTE == 21, f"got {op.TARGET_MAX_DTE}")
check("ideal is the midpoint", op.TARGET_IDEAL_DTE == 17, f"got {op.TARGET_IDEAL_DTE}")
check("label reads 14-21 DTE", op.DTE_LABEL == "14-21 DTE", f"got {op.DTE_LABEL}")

print("\n2. A short window is honoured")
op = reload_with(OPTION_MIN_DTE=0, OPTION_MAX_DTE=2)
check("min is 0", op.TARGET_MIN_DTE == 0)
check("max is 2", op.TARGET_MAX_DTE == 2)
check("ideal recomputed", op.TARGET_IDEAL_DTE == 1, f"got {op.TARGET_IDEAL_DTE}")
check("fallback floor cannot go negative", op.FALLBACK_FLOOR_DTE == 0,
      f"got {op.FALLBACK_FLOOR_DTE}")
check("label follows", op.DTE_LABEL == "0-2 DTE", f"got {op.DTE_LABEL}")

print("\n3. Alert headers follow the configured window, not a stale string")
import src.utils.telegram_formatter as fmt
importlib.reload(fmt)
from src.models.signal import TradeSignal

sig = TradeSignal(ticker="PLTR", price=150.0, signal_direction="Long",
                  strategy_type="Volume Breakout", timestamp="2026-09-03 10:00:00",
                  conviction="⚪ UNSCORED", win_probability=0.36, stop_loss=145.0,
                  take_profit=160.0, catalyst="t", historical_edge="n/a",
                  context_score="x")
msg = fmt.format_telegram_alert(sig)
check("header shows the configured window", "0-2 DTE" in msg,
      "header still says something else")
check("header does not show the old hardcoded window", "14-21 DTE" not in msg)

print("\n4. Inverted bounds are corrected, not obeyed")
op = reload_with(OPTION_MIN_DTE=30, OPTION_MAX_DTE=7)
check("bounds swapped into order", (op.TARGET_MIN_DTE, op.TARGET_MAX_DTE) == (7, 30),
      f"got {(op.TARGET_MIN_DTE, op.TARGET_MAX_DTE)}")

print("\n5. Malformed configuration falls back to the default")
op = reload_with(OPTION_MIN_DTE="soon", OPTION_MAX_DTE=-5)
check("non-numeric ignored", op.TARGET_MIN_DTE == 14, f"got {op.TARGET_MIN_DTE}")
check("negative ignored", op.TARGET_MAX_DTE == 21, f"got {op.TARGET_MAX_DTE}")

print("\n6. The pipeline no longer hardcodes the window")
op = reload_with()
import inspect

import src.execution.signal_pipeline as sp
importlib.reload(sp)
src = inspect.getsource(sp.SignalPipeline.process_signal)
check("no hardcoded min_dte=14 at the call site", "min_dte=14" not in src)
check("calls get_target_expiration without a window", "get_target_expiration(ticker, session=" in src)

# leave the environment as we found it
for k in ("OPTION_MIN_DTE", "OPTION_MAX_DTE"):
    os.environ.pop(k, None)

print("\n" + "=" * 62)
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
