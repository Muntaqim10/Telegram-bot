"""Manual test for RiskManager.check_expiration_warnings().

Verifies the anti-worthless-expiry guard: a position whose option is about to
expire warns exactly once, and the expiration_warned flag stops it re-firing.

Run: python scripts/manual_test_expiration_warnings.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.risk_manager import RiskManager

MOCK_TODAY = "2026-09-02"          # mocked "current_date"
EXPIRES_TOMORROW = "2026-09-03"    # 1 day out  -> should warn
EXPIRES_IN_A_WEEK = "2026-09-09"   # 7 days out -> should stay quiet
EXPIRED_YESTERDAY = "2026-09-01"   # already gone -> should be pruned

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def add(rm, *args, **kwargs):
    """add_position, confirmed by default.

    Only confirmed positions receive alerts -- the bot cannot see the brokerage
    account, so an unconfirmed alert is a suggestion, not a holding. The suites below
    exercise the alerting, so they take the trade; the gate itself is tested separately.
    """
    kwargs.setdefault("confirmed", True)
    return rm.add_position(*args, **kwargs)


rm = RiskManager(state_path=None)

print("\n1. Position expiring in 1 day (via add_position)")
add(rm, 
    "AAPL", entry_price=230.0, initial_atr=2.0, direction="Long",
    option_expiration=EXPIRES_TOMORROW,
)
check("option_expiration stored on the position dict",
      rm.active_positions["AAPL"]["option_expiration"] == EXPIRES_TOMORROW)
check("expiration_warned starts False",
      rm.active_positions["AAPL"]["expiration_warned"] is False)

print("\n2. Position expiring in 7 days (should never warn)")
add(rm, 
    "MSFT", entry_price=500.0, initial_atr=3.0, direction="Long",
    option_expiration=EXPIRES_IN_A_WEEK,
)

print("\n3. Position that gets its expiration from the live pipeline path")
add(rm, "NVDA", entry_price=180.0, initial_atr=4.0, direction="Long")
check("no expiration before attach_option_pricing",
      rm.active_positions["NVDA"]["option_expiration"] is None)
rm.attach_option_pricing(
    "NVDA", option_entry_price=8.10, option_entry_delta=0.75,
    option_entry_theta=-0.12, option_expiration=EXPIRES_TOMORROW,
)
check("attach_option_pricing() sets option_expiration",
      rm.active_positions["NVDA"]["option_expiration"] == EXPIRES_TOMORROW)

print(f"\n4. First check_expiration_warnings('{MOCK_TODAY}')")
first = rm.check_expiration_warnings(MOCK_TODAY)
print(f"     returned: {first}")
warned = sorted(w["ticker"] for w in first)
check("warns for both 1-day positions, not the 7-day one",
      warned == ["AAPL", "NVDA"], f"got {warned}")
check("days_to_expiration == 1",
      all(w["days_to_expiration"] == 1 for w in first))
check("expiration_warned flipped to True on AAPL",
      rm.active_positions["AAPL"]["expiration_warned"] is True)
check("MSFT (7 DTE) left unwarned",
      rm.active_positions["MSFT"]["expiration_warned"] is False)

print(f"\n5. Second check_expiration_warnings('{MOCK_TODAY}') -- same date, no duplicates")
second = rm.check_expiration_warnings(MOCK_TODAY)
print(f"     returned: {second}")
check("returns no duplicate warnings", second == [], f"got {second}")

print("\n6. reset_daily() drops expired/intraday positions but CARRIES live options forward")
rm2 = RiskManager(state_path=None)
add(rm2, "TSLA", entry_price=400.0, initial_atr=5.0, direction="Long",
                 option_expiration=EXPIRED_YESTERDAY)   # expired -> drop
add(rm2, "AMD", entry_price=160.0, initial_atr=2.0, direction="Long",
                 option_expiration=EXPIRES_TOMORROW)    # live    -> keep
add(rm2, "F", entry_price=12.0, initial_atr=0.3, direction="Long")  # intraday -> drop
rm2.active_positions["AMD"]["expiration_warned"] = True
rm2.reset_daily(current_date=MOCK_TODAY)
check("live option position carried across the daily reset",
      sorted(rm2.active_positions) == ["AMD"], f"got {sorted(rm2.active_positions)}")
check("carried position can warn again today",
      rm2.active_positions["AMD"]["expiration_warned"] is False)
check("carried position still warns after the reset",
      [w["ticker"] for w in rm2.check_expiration_warnings(MOCK_TODAY)] == ["AMD"])

print("\n6b. reset_daily() with no date keeps the original clear-everything behaviour")
rm2b = RiskManager(state_path=None)
add(rm2b, "AMD", entry_price=160.0, initial_atr=2.0, direction="Long",
                  option_expiration=EXPIRES_TOMORROW)
rm2b.reset_daily()
check("no current_date -> everything cleared", rm2b.active_positions == {})

print("\n7. Position with no expiration is skipped, not crashed on")
rm3 = RiskManager(state_path=None)
add(rm3, "SPY", entry_price=660.0, initial_atr=3.0, direction="Long")
check("no warning for a position without an expiration",
      rm3.check_expiration_warnings(MOCK_TODAY) == [])

print("\n8. A warning handed back after a failed send re-fires on the next sweep")
rm4 = RiskManager(state_path=None)
add(rm4, "META", entry_price=700.0, initial_atr=5.0, direction="Long",
                 option_expiration=EXPIRES_TOMORROW)
check("warns on the first sweep",
      [w["ticker"] for w in rm4.check_expiration_warnings(MOCK_TODAY)] == ["META"])
check("stays suppressed while the flag stands",
      rm4.check_expiration_warnings(MOCK_TODAY) == [])
# This is exactly what main._dispatch_expiration_warnings() does when
# dispatch_informational() returns False -- it hands the warning back.
rm4.active_positions["META"]["expiration_warned"] = False
retry = rm4.check_expiration_warnings(MOCK_TODAY)
check("re-fires once the flag is handed back",
      [w["ticker"] for w in retry] == ["META"], f"got {retry}")

print("\n9. Positions survive a process restart (persisted to disk and reloaded)")
import tempfile
state_file = os.path.join(tempfile.mkdtemp(), "active_positions.json")

rm5 = RiskManager(state_path=state_file)
add(rm5, "GOOG", entry_price=250.0, initial_atr=3.0, direction="Long",
                 option_expiration=EXPIRES_TOMORROW)
rm5.attach_option_pricing("GOOG", option_entry_price=6.40, option_entry_delta=0.75,
                          option_entry_theta=-0.09, option_expiration=EXPIRES_TOMORROW)
check("state file written on add", os.path.exists(state_file))

# A brand-new RiskManager stands in for the process having restarted.
rm6 = RiskManager(state_path=state_file)
check("nothing in memory before load", rm6.active_positions == {})
restored = rm6.load_positions(current_date=MOCK_TODAY)
check("one position restored", restored == 1, f"got {restored}")
check("expiration survived the round trip",
      rm6.active_positions["GOOG"]["option_expiration"] == EXPIRES_TOMORROW)
check("entry price survived the round trip",
      rm6.active_positions["GOOG"]["entry_price"] == 250.0)
check("restored position still warns",
      [w["ticker"] for w in rm6.check_expiration_warnings(MOCK_TODAY)] == ["GOOG"])

# An expired contract must not come back from the dead.
rm7 = RiskManager(state_path=state_file)
add(rm7, "INTC", entry_price=25.0, initial_atr=0.5, direction="Long",
                 option_expiration=EXPIRED_YESTERDAY)
rm8 = RiskManager(state_path=state_file)
rm8.load_positions(current_date=MOCK_TODAY)
check("expired position dropped on load", "INTC" not in rm8.active_positions,
      f"got {sorted(rm8.active_positions)}")

# A corrupt state file must not take the bot down.
with open(state_file, "w", encoding="utf-8") as f:
    f.write("{not valid json")
rm9 = RiskManager(state_path=state_file)
check("corrupt state file degrades to empty, no crash",
      rm9.load_positions(current_date=MOCK_TODAY) == 0)

print("\n" + "=" * 60)
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
