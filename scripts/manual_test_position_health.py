"""Manual test for RiskManager.check_position_health().

Replays the two exit failures visible in the 2019-2026 Robinhood history:
  - holding past day 5, where the account's win rate falls from ~50% to 27%
  - riding a contract down past -50% of premium instead of cutting it

Run: python scripts/manual_test_position_health.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.risk_manager import RiskManager

TODAY = "2026-09-02"
failures = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def new_rm():
    return RiskManager(state_path=None)


print("\n1. TIME STOP -- the INTC 11/21/2025 pattern (bought, held 25 days, expired)")
rm = new_rm()
rm.add_position("INTC", entry_price=40.0, initial_atr=1.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-08-26")  # 7 days held
a = rm.check_position_health(TODAY)
check("fires a TIME_STOP", [x["reason"] for x in a] == ["TIME_STOP"], f"got {[x['reason'] for x in a]}")
check("reports the right hold length", a[0]["days_held"] == 7, f"got {a[0]['days_held']}")
check("does not re-fire the same day", rm.check_position_health(TODAY) == [])

print("\n2. A position inside the 5-day window stays quiet")
rm = new_rm()
rm.add_position("AAPL", entry_price=230.0, initial_atr=2.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-08-31")  # 2 days held
check("no alert at 2 days held", rm.check_position_health(TODAY) == [])

print("\n3. PREMIUM STOP -- the SOFI/GLD pattern (premium halved, still holding)")
rm = new_rm()
rm.add_position("GLD", entry_price=327.0, initial_atr=3.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-09-01")
rm.attach_option_pricing("GLD", option_entry_price=4.00, option_entry_delta=0.75,
                         option_entry_theta=-0.10, option_expiration="2026-09-18")
# Stock drops $3.00 -> delta-approx premium 4.00 - 2.25 = 1.75 => -56%
rm.update_trailing_stop("GLD", current_price=324.0, current_atr=3.0, direction="Long")
a = rm.check_position_health(TODAY)
check("fires a PREMIUM_STOP", [x["reason"] for x in a] == ["PREMIUM_STOP"], f"got {[x['reason'] for x in a]}")
check("estimated loss is past -50%", a[0]["est_option_pnl_pct"] <= -0.50,
      f"got {a[0]['est_option_pnl_pct']:.3f}")
check("does not re-fire the same day", rm.check_position_health(TODAY) == [])

print("\n4. A modest drawdown does not trip the premium stop")
rm = new_rm()
rm.add_position("NVDA", entry_price=180.0, initial_atr=2.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-09-01")
rm.attach_option_pricing("NVDA", option_entry_price=8.00, option_entry_delta=0.75,
                         option_entry_theta=-0.10, option_expiration="2026-09-18")
rm.update_trailing_stop("NVDA", current_price=178.0, current_atr=2.0, direction="Long")  # ~-19%
check("no alert on a -19% estimate", rm.check_position_health(TODAY) == [])

print("\n5. Both alerts can fire for the same position")
rm = new_rm()
rm.add_position("PLTR", entry_price=150.0, initial_atr=2.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-08-20")  # 13 days
rm.attach_option_pricing("PLTR", option_entry_price=5.00, option_entry_delta=0.75,
                         option_entry_theta=-0.10, option_expiration="2026-09-18")
rm.update_trailing_stop("PLTR", current_price=145.0, current_atr=2.0, direction="Long")
reasons = sorted(x["reason"] for x in rm.check_position_health(TODAY))
check("both TIME_STOP and PREMIUM_STOP fire", reasons == ["PREMIUM_STOP", "TIME_STOP"], f"got {reasons}")

print("\n6. A position with no option pricing never trips the premium stop")
rm = new_rm()
rm.add_position("SPY", entry_price=660.0, initial_atr=3.0, direction="Long",
                entry_date="2026-09-01")
rm.update_trailing_stop("SPY", current_price=600.0, current_atr=3.0, direction="Long")
check("no premium alert without option data",
      [x["reason"] for x in rm.check_position_health(TODAY)] == [])

print("\n7. Short direction is valued the right way round")
rm = new_rm()
rm.add_position("AMD", entry_price=165.0, initial_atr=2.0, direction="Short",
                option_expiration="2026-09-18", entry_date="2026-09-01")
rm.attach_option_pricing("AMD", option_entry_price=4.00, option_entry_delta=-0.75,
                         option_entry_theta=-0.10, option_expiration="2026-09-18")
rm.update_trailing_stop("AMD", current_price=169.0, current_atr=2.0, direction="Short")  # against a put
a = rm.check_position_health(TODAY)
check("put loses value when the stock rises",
      [x["reason"] for x in a] == ["PREMIUM_STOP"], f"got {[x['reason'] for x in a]}")

print("\n8. reset_daily clears the flags so an open position warns again tomorrow")
rm = new_rm()
rm.add_position("INTC", entry_price=40.0, initial_atr=1.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-08-26")
rm.check_position_health(TODAY)
check("suppressed while flagged", rm.check_position_health(TODAY) == [])
rm.reset_daily(current_date=TODAY)
check("position carried across the reset", "INTC" in rm.active_positions)
check("warns again after the reset",
      [x["reason"] for x in rm.check_position_health(TODAY)] == ["TIME_STOP"])

print("\n9. A failed dispatch re-queues the alert (what main.py does on send failure)")
rm = new_rm()
rm.add_position("META", entry_price=700.0, initial_atr=5.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-08-26")
rm.check_position_health(TODAY)
check("consumed after firing", rm.check_position_health(TODAY) == [])
rm.requeue_warning("META", "time_stop_warned")
check("re-fires after requeue",
      [x["reason"] for x in rm.check_position_health(TODAY)] == ["TIME_STOP"])

print("\n10. Theta decay is priced in, not ignored (review finding #2)")
rm = new_rm()
rm.add_position("MSFT", entry_price=500.0, initial_atr=3.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-08-26")  # 7 days held
rm.attach_option_pricing("MSFT", option_entry_price=5.00, option_entry_delta=0.75,
                         option_entry_theta=-0.30, option_expiration="2026-09-18")
rm.update_trailing_stop("MSFT", current_price=500.0, current_atr=3.0, direction="Long")  # stock flat
est = rm._estimate_option_pnl(rm.active_positions["MSFT"], 500.0, TODAY)
check("a flat stock still shows a loss after 7 days of theta", est < -0.35, f"got {est:.3f}")
check("-42% is correctly still short of the -50% line",
      "PREMIUM_STOP" not in [x["reason"] for x in rm.check_position_health(TODAY)])

# Same contract, same flat stock, three more days of decay -> theta alone trips the stop.
rm = new_rm()
rm.add_position("MSFT", entry_price=500.0, initial_atr=3.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-08-23")  # 10 days held
rm.attach_option_pricing("MSFT", option_entry_price=5.00, option_entry_delta=0.75,
                         option_entry_theta=-0.30, option_expiration="2026-09-18")
rm.update_trailing_stop("MSFT", current_price=500.0, current_atr=3.0, direction="Long")
check("theta alone trips the premium stop by day 10",
      "PREMIUM_STOP" in [x["reason"] for x in rm.check_position_health(TODAY)])

print("\n11. A stale price is not used to fire a stop (review findings #1/#3)")
rm = new_rm()
rm.add_position("UBER", entry_price=77.0, initial_atr=1.0, direction="Long",
                option_expiration="2026-09-18", entry_date="2026-08-26")
rm.attach_option_pricing("UBER", option_entry_price=4.00, option_entry_delta=0.75,
                         option_entry_theta=-0.05, option_expiration="2026-09-18")
rm.update_trailing_stop("UBER", current_price=70.0, current_atr=1.0, direction="Long")
rm.active_positions["UBER"]["last_price_ts"] = 0  # simulate a ticker rotated out of the stream
alerts = rm.check_position_health(TODAY)
check("no PREMIUM_STOP fired off a stale price",
      "PREMIUM_STOP" not in [x["reason"] for x in alerts], f"got {[x['reason'] for x in alerts]}")
check("TIME_STOP still fires, with no estimate attached",
      [x["reason"] for x in alerts] == ["TIME_STOP"] and alerts[0]["est_option_pnl_pct"] is None)

print("\n12. An unreadable expiration keeps the position instead of dropping it (finding #4)")
rm = new_rm()
rm.add_position("CVX", entry_price=230.0, initial_atr=2.0, direction="Long",
                option_expiration="2026-09-18T00:00:00", entry_date="2026-08-26")  # bad format
rm.reset_daily(current_date=TODAY)
check("position survives an unparseable expiration", "CVX" in rm.active_positions,
      f"got {sorted(rm.active_positions)}")

print("\n13. load_positions refuses to clobber live positions (finding #7)")
import tempfile
sf = os.path.join(tempfile.mkdtemp(), "state.json")
seed = RiskManager(state_path=sf)
seed.add_position("GOOG", entry_price=250.0, initial_atr=3.0, direction="Long",
                  option_expiration="2026-09-18")
live = RiskManager(state_path=sf)
live.add_position("AMD", entry_price=160.0, initial_atr=2.0, direction="Long",
                  option_expiration="2026-09-18")
check("load refuses when positions are already in memory",
      live.load_positions(current_date=TODAY) == 0)
check("the live position was not discarded", "AMD" in live.active_positions)

print("\n" + "=" * 60)
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
