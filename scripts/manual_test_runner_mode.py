"""Manual test for runner mode and the take-profit ceiling.

The scenario: a stock soars, an OTM strike goes ITM, and it keeps going. With a fixed
take-profit the position closes at roughly 1.7x daily ATR while the downside stays open
to the whole premium -- which truncates exactly the right tail that makes owning a long
option worth anything.

Runner mode turns the target into a trigger instead of an exit.

Run: python scripts/manual_test_runner_mode.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.risk_manager import RiskManager

failures = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def position(runner, direction="Long", entry=100.0, tp=110.0):
    rm = RiskManager(state_path=None, runner_mode=runner, runner_trail_atr=1.5)
    rm.add_position("SOAR", entry_price=entry, initial_atr=2.0, direction=direction,
                    stop_loss=(entry - 5 if direction == "Long" else entry + 5),
                    take_profit=tp, option_expiration="2026-12-18", confirmed=True)
    rm.attach_option_pricing("SOAR", option_entry_price=2.00, option_entry_delta=0.40,
                             option_entry_theta=-0.05, option_expiration="2026-12-18")
    return rm


ATR = 2.0

print("\n1. Runner mode OFF -- the target closes the position (previous behaviour)")
rm = position(runner=False)
out = rm.update_trailing_stop("SOAR", current_price=112.0, current_atr=ATR, direction="Long")
check("status is TP_HIT", out["status"] == "TP_HIT", f"got {out['status']}")

print("\n2. Runner mode ON -- the target starts a trail instead of closing")
rm = position(runner=True)
out = rm.update_trailing_stop("SOAR", current_price=112.0, current_atr=ATR, direction="Long")
check("status is TP_HIT_TRAILING", out["status"] == "TP_HIT_TRAILING", f"got {out['status']}")
check("position stays open", "SOAR" in rm.active_positions)
check("marked as a runner", rm.active_positions["SOAR"]["runner"] is True)
check("trail set 1.5 ATR below price",
      abs(rm.active_positions["SOAR"]["trailing_stop"] - (112.0 - 3.0)) < 1e-9,
      f"got {rm.active_positions['SOAR']['trailing_stop']}")
check("the alert carries the trail level", "trailing_stop" in out)

print("\n3. The stock keeps soaring -- the trail follows, nothing closes")
for price in (120.0, 135.0, 150.0):
    out = rm.update_trailing_stop("SOAR", current_price=price, current_atr=ATR, direction="Long")
    check(f"still open at ${price:.0f}", out["status"] == "ACTIVE", f"got {out['status']}")
check("trail ratcheted up to 147.00",
      abs(rm.active_positions["SOAR"]["trailing_stop"] - 147.0) < 1e-9,
      f"got {rm.active_positions['SOAR']['trailing_stop']}")
check("the target does NOT re-trigger", rm.active_positions["SOAR"]["runner"] is True)

print("\n4. The move finally rolls over -- the trail closes it, far above the target")
out = rm.update_trailing_stop("SOAR", current_price=146.0, current_atr=ATR, direction="Long")
check("status is RUNNER_STOPPED", out["status"] == "RUNNER_STOPPED", f"got {out['status']}")
check("exit is well above the 110 target", out["exit_price"] == 146.0)
check("captured +46% on the underlying instead of +12%",
      abs(out["pnl"] - 0.46) < 1e-9, f"got {out['pnl']:.4f}")

print("\n5. The trail never loosens on a pullback")
rm = position(runner=True)
rm.update_trailing_stop("SOAR", current_price=112.0, current_atr=ATR, direction="Long")
rm.update_trailing_stop("SOAR", current_price=140.0, current_atr=ATR, direction="Long")
high_trail = rm.active_positions["SOAR"]["trailing_stop"]
rm.update_trailing_stop("SOAR", current_price=138.0, current_atr=ATR, direction="Long")
check("a lower price does not lower the stop",
      rm.active_positions["SOAR"]["trailing_stop"] == high_trail,
      f"{rm.active_positions['SOAR']['trailing_stop']} vs {high_trail}")

print("\n6. Short side behaves symmetrically")
rm = position(runner=True, direction="Short", entry=100.0, tp=90.0)
out = rm.update_trailing_stop("SOAR", current_price=88.0, current_atr=ATR, direction="Short")
check("target triggers the trail", out["status"] == "TP_HIT_TRAILING", f"got {out['status']}")
check("trail set 1.5 ATR above price",
      abs(rm.active_positions["SOAR"]["trailing_stop"] - 91.0) < 1e-9,
      f"got {rm.active_positions['SOAR']['trailing_stop']}")
out = rm.update_trailing_stop("SOAR", current_price=70.0, current_atr=ATR, direction="Short")
check("keeps running down", out["status"] == "ACTIVE", f"got {out['status']}")
check("trail follows down to 73.00",
      abs(rm.active_positions["SOAR"]["trailing_stop"] - 73.0) < 1e-9,
      f"got {rm.active_positions['SOAR']['trailing_stop']}")
out = rm.update_trailing_stop("SOAR", current_price=74.0, current_atr=ATR, direction="Short")
check("trail closes it", out["status"] == "RUNNER_STOPPED", f"got {out['status']}")

print("\n7. A normal stop-out still works with runner mode on")
rm = position(runner=True)
out = rm.update_trailing_stop("SOAR", current_price=94.0, current_atr=ATR, direction="Long")
check("stopped out below the entry stop", out["status"] == "STOPPED_OUT", f"got {out['status']}")

print("\n8. The time stop must not nag someone out of a winner")
rm = RiskManager(state_path=None)
rm.add_position("WIN", entry_price=100.0, initial_atr=2.0, direction="Long",
                option_expiration="2026-12-18", entry_date="2026-08-20", confirmed=True)
rm.attach_option_pricing("WIN", option_entry_price=2.00, option_entry_delta=0.40,
                         option_entry_theta=-0.01, option_expiration="2026-12-18")
rm.update_trailing_stop("WIN", current_price=130.0, current_atr=2.0, direction="Long")
alerts = rm.check_position_health("2026-09-02")   # 13 days held, deeply profitable
check("no TIME_STOP on a winning position",
      "TIME_STOP" not in [a["reason"] for a in alerts], f"got {[a['reason'] for a in alerts]}")

rm2 = RiskManager(state_path=None)
rm2.add_position("LOSE", entry_price=100.0, initial_atr=2.0, direction="Long",
                 option_expiration="2026-12-18", entry_date="2026-08-20", confirmed=True)
rm2.attach_option_pricing("LOSE", option_entry_price=2.00, option_entry_delta=0.40,
                          option_entry_theta=-0.01, option_expiration="2026-12-18")
rm2.update_trailing_stop("LOSE", current_price=98.0, current_atr=2.0, direction="Long")
check("TIME_STOP still fires on a losing one held too long",
      "TIME_STOP" in [a["reason"] for a in rm2.check_position_health("2026-09-02")])

print("\n9. Runner state survives a restart")
import tempfile
sf = os.path.join(tempfile.mkdtemp(), "state.json")
rm = RiskManager(state_path=sf, runner_mode=True, runner_trail_atr=1.5)
rm.add_position("SOAR", entry_price=100.0, initial_atr=2.0, direction="Long",
                stop_loss=95.0, take_profit=110.0, option_expiration="2026-12-18",
                confirmed=True)
rm.update_trailing_stop("SOAR", current_price=112.0, current_atr=ATR, direction="Long")
rm2 = RiskManager(state_path=sf, runner_mode=True, runner_trail_atr=1.5)
rm2.load_positions()
check("runner flag persisted", rm2.active_positions["SOAR"]["runner"] is True)
check("trail level persisted",
      abs(rm2.active_positions["SOAR"]["trailing_stop"] - 109.0) < 1e-9,
      f"got {rm2.active_positions['SOAR']['trailing_stop']}")

print("\n" + "=" * 62)
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
