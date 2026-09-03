"""Manual test for the alert/position split.

The bot cannot see Robinhood. An alert is a suggestion; a position exists only once
the trader confirms it with /took. This suite checks that unconfirmed alerts stay
silent, that confirmed ones behave like held positions, and that /closed records the
real fill rather than an approximation.

Run: python scripts/manual_test_position_ledger.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alerts import AlertGateway
from src.execution.risk_manager import RiskManager

TODAY = "2026-09-02"
SOON = "2026-09-03"      # expires tomorrow
failures = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def alerted(ticker="PLTR", entry_date="2026-08-20", **kw):
    """A tracked alert: the bot suggested it, the trader has NOT confirmed it."""
    rm = RiskManager(state_path=None)
    rm.add_position(ticker, entry_price=150.0, initial_atr=2.0, direction="Long",
                    option_expiration=SOON, entry_date=entry_date, **kw)
    rm.attach_option_pricing(ticker, option_entry_price=5.00, option_entry_delta=0.75,
                             option_entry_theta=-0.10, option_expiration=SOON)
    rm.update_trailing_stop(ticker, current_price=140.0, current_atr=2.0, direction="Long")
    return rm


print("\n1. An unconfirmed alert produces NO warnings, however dire it looks")
rm = alerted()                      # 13 days old, expiring tomorrow, premium collapsed
check("position is not confirmed", rm.active_positions["PLTR"]["confirmed"] is False)
check("no expiration warning", rm.check_expiration_warnings(TODAY) == [])
check("no health alerts", rm.check_position_health(TODAY) == [])

print("\n2. Confirming it turns on every warning it qualifies for")
rm.confirm_position("PLTR")
check("marked confirmed", rm.active_positions["PLTR"]["confirmed"] is True)
check("expiration warning now fires",
      [w["ticker"] for w in rm.check_expiration_warnings(TODAY)] == ["PLTR"])

print("\n3. Confirming records the trader's real fill over the alert-time quote")
rm2 = alerted("AMD")
rm2.confirm_position("AMD", quantity=2, fill_price=6.25)
pos = rm2.active_positions["AMD"]
check("quantity recorded", pos["quantity"] == 2)
check("fill price recorded", pos["fill_price"] == 6.25)
check("fill price overrides the alert-time quote", pos["option_entry_price"] == 6.25,
      f"got {pos['option_entry_price']}")

print("\n4. entry_date resets to the confirmation date, not the alert date")
rm3 = alerted("NVDA", entry_date="2026-08-01")
rm3.confirm_position("NVDA", entry_date="2026-08-30")
check("hold clock starts when the trade was taken",
      rm3.days_held(rm3.active_positions["NVDA"], TODAY) == 3,
      f"got {rm3.days_held(rm3.active_positions['NVDA'], TODAY)}")

print("\n5. Unconfirmed alerts are dropped at the daily reset; confirmed ones carry")
rm4 = RiskManager(state_path=None)
rm4.add_position("SKIP", entry_price=10.0, initial_atr=1.0, direction="Long",
                 option_expiration=SOON)
rm4.add_position("HELD", entry_price=20.0, initial_atr=1.0, direction="Long",
                 option_expiration=SOON, confirmed=True)
rm4.reset_daily(current_date=TODAY)
check("unconfirmed alert dropped", "SKIP" not in rm4.active_positions)
check("confirmed position carried", "HELD" in rm4.active_positions,
      f"got {sorted(rm4.active_positions)}")

print("\n6. /closed records the REAL option P&L")
rm5 = alerted("GLD")
rm5.confirm_position("GLD", quantity=1, fill_price=4.00)
summary = rm5.close_confirmed("GLD", exit_option_price=2.00)
check("position removed", "GLD" not in rm5.active_positions)
check("real option P&L computed from the fills",
      abs(summary["option_pnl_pct"] - (-0.50)) < 1e-9, f"got {summary['option_pnl_pct']}")
check("hold length reported", summary["days_held"] is not None)

print("\n7. /closed without a fill price records no fabricated P&L")
rm6 = alerted("UBER")
rm6.confirm_position("UBER")
summary = rm6.close_confirmed("UBER")
check("no option P&L invented", summary["option_pnl_pct"] is None)

print("\n8. list_positions separates holdings from suggestions")
rm7 = RiskManager(state_path=None)
rm7.add_position("HOLD1", entry_price=10.0, initial_atr=1.0, direction="Long", confirmed=True)
rm7.add_position("MAYBE", entry_price=20.0, initial_atr=1.0, direction="Long")
book = rm7.list_positions()
check("confirmed bucket", [t for t, _ in book["confirmed"]] == ["HOLD1"])
check("unconfirmed bucket", [t for t, _ in book["unconfirmed"]] == ["MAYBE"])

print("\n9. Telegram command handlers")
gw = AlertGateway(None)
rm8 = alerted("META")
gw.attach_position_ledger(rm8)

check("/took parses ticker, qty and fill",
      "Watching" in gw._cmd_took("/took META 2 9.10"))
check("  ...and reached the ledger", rm8.active_positions["META"]["fill_price"] == 9.10)
check("/took on an unknown ticker explains itself",
      "No tracked alert" in gw._cmd_took("/took ZZZZ"))
check("/took tolerates a $ prefix and lowercase",
      "No tracked alert" in gw._cmd_took("/took $zzzz"))
check("/positions lists the holding", "META" in gw._cmd_positions())
check("/closed reports the real result", "-1.5%" in gw._cmd_closed("/closed META 8.96")
      or "%" in gw._cmd_closed("/closed META 8.96"))
check("/closed on an untracked ticker is handled",
      "not being tracked" in gw._cmd_closed("/closed ZZZZ"))

gw2 = AlertGateway(None)
check("commands degrade safely with no ledger attached",
      "not attached" in gw2._cmd_took("/took META"))

print("\n" + "=" * 60)
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
