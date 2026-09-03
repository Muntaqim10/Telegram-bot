"""Manual test for the risk budget circuit breaker.

This is the one component whose job is to say no. If it fails open, it fails silently
and the trader finds out from their balance, so every path is checked explicitly.

Run: python scripts/manual_test_risk_budget.py
"""
import datetime
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.risk_budget import RiskBudget

TODAY = datetime.date(2026, 9, 3)      # a Thursday; week starts Monday 2026-08-31
failures = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def db_with(*fills):
    """A throwaway database holding the given (close_date, pnl) closed contracts."""
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE real_fills (
        id INTEGER PRIMARY KEY AUTOINCREMENT, contract TEXT, ticker TEXT, close_date TEXT,
        pnl REAL)""")
    for close_date, pnl in fills:
        conn.execute("INSERT INTO real_fills (contract, ticker, close_date, pnl) VALUES (?,?,?,?)",
                     (f"X {close_date}", "X", close_date, pnl))
    conn.commit()
    conn.close()
    return path


print("\n1. With no limits set, the bot never blocks")
b = RiskBudget(db_path=db_with(("2026-09-02", -5000.0)),
               max_loss_week=0, max_loss_month=0, max_premium_per_trade=0, max_open_positions=0)
g = b.check_entry(premium=99999.0, open_positions=99, today=TODAY)
check("entry allowed", g["allowed"] is True, g["reason"] or "")
check("status says no limits configured", b.status(TODAY)["any_limit_set"] is False)
check("describe() explains how to turn it on", "No limits set" in b.describe(TODAY))

print("\n2. Weekly loss ceiling")
b = RiskBudget(db_path=db_with(("2026-09-01", -300.0), ("2026-09-02", -250.0)),
               max_loss_week=500, max_loss_month=0, max_premium_per_trade=0, max_open_positions=0)
st = b.status(TODAY)
check("week P&L summed from closed contracts", st["week_pnl"] == -550.0, f"got {st['week_pnl']}")
check("breach detected", st["halted"] is True)
check("entry blocked", b.check_entry(today=TODAY)["allowed"] is False)
check("reason names the limit", "weekly loss" in b.check_entry(today=TODAY)["reason"])

print("\n3. Losses from BEFORE this week do not count against it")
b = RiskBudget(db_path=db_with(("2026-08-20", -5000.0)),
               max_loss_week=500, max_loss_month=0, max_premium_per_trade=0, max_open_positions=0)
check("last week's loss excluded", b.status(TODAY)["week_pnl"] == 0.0,
      f"got {b.status(TODAY)['week_pnl']}")
check("entry still allowed", b.check_entry(today=TODAY)["allowed"] is True)

print("\n4. Monthly ceiling is independent of the weekly one")
b = RiskBudget(db_path=db_with(("2026-09-01", -400.0), ("2026-09-02", -400.0)),
               max_loss_week=0, max_loss_month=700, max_premium_per_trade=0, max_open_positions=0)
check("month P&L summed", b.status(TODAY)["month_pnl"] == -800.0)
check("monthly breach halts entries", b.check_entry(today=TODAY)["allowed"] is False)
check("reason names the monthly limit", "monthly loss" in b.check_entry(today=TODAY)["reason"])

print("\n5. Wins offset losses inside the period")
b = RiskBudget(db_path=db_with(("2026-09-01", -600.0), ("2026-09-02", 400.0)),
               max_loss_week=500, max_loss_month=0, max_premium_per_trade=0, max_open_positions=0)
check("net is used, not gross losses", b.status(TODAY)["week_pnl"] == -200.0)
check("entry allowed while net is inside the limit",
      b.check_entry(today=TODAY)["allowed"] is True)

print("\n6. Per-trade premium cap -- the pattern behind every worst loss")
b = RiskBudget(db_path=db_with(), max_loss_week=0, max_loss_month=0,
               max_premium_per_trade=500, max_open_positions=0)
check("a $400 entry passes", b.check_entry(premium=400.0, today=TODAY)["allowed"] is True)
check("a $3,350 entry is blocked (the SOFI-sized trade)",
      b.check_entry(premium=3350.0, today=TODAY)["allowed"] is False)
check("reason quotes both numbers",
      "3,350" in b.check_entry(premium=3350.0, today=TODAY)["reason"])

print("\n7. Concurrency cap")
b = RiskBudget(db_path=db_with(), max_loss_week=0, max_loss_month=0,
               max_premium_per_trade=0, max_open_positions=3)
check("under the cap is fine", b.check_entry(open_positions=2, today=TODAY)["allowed"] is True)
check("at the cap blocks", b.check_entry(open_positions=3, today=TODAY)["allowed"] is False)

print("\n8. Open positions do not count toward realized loss")
path = db_with(("2026-09-01", -900.0))
conn = sqlite3.connect(path)
conn.execute("INSERT INTO real_fills (contract, ticker, close_date, pnl) VALUES (?,?,?,?)",
             ("OPEN", "Y", None, None))
conn.commit()
conn.close()
b = RiskBudget(db_path=path, max_loss_week=1000, max_loss_month=0,
               max_premium_per_trade=0, max_open_positions=0)
check("unclosed contracts excluded", b.status(TODAY)["week_pnl"] == -900.0,
      f"got {b.status(TODAY)['week_pnl']}")

print("\n9. The halt is announced once per day, not on every blocked signal")
b = RiskBudget(db_path=db_with(("2026-09-02", -900.0)),
               max_loss_week=500, max_loss_month=0, max_premium_per_trade=0, max_open_positions=0)
check("first blocked signal announces", b.should_announce_halt(TODAY) is True)
check("second does not", b.should_announce_halt(TODAY) is False)
check("a new day announces again",
      b.should_announce_halt(TODAY + datetime.timedelta(days=1)) is True)

print("\n10. A missing or unreadable database fails SAFE, not open")
b = RiskBudget(db_path="/nonexistent/path/to.db", max_loss_week=500, max_loss_month=0,
               max_premium_per_trade=0, max_open_positions=0)
check("no crash", b.status(TODAY)["week_pnl"] == 0.0)
check("per-trade cap still enforced without any P&L history",
      RiskBudget(db_path="/nope.db", max_loss_week=0, max_loss_month=0,
                 max_premium_per_trade=100, max_open_positions=0)
      .check_entry(premium=5000.0, today=TODAY)["allowed"] is False)

print("\n11. Environment configuration")
os.environ["RISK_MAX_LOSS_PER_WEEK"] = "$1,250"
os.environ["RISK_MAX_PREMIUM_PER_TRADE"] = "not a number"
b = RiskBudget(db_path=db_with())
check("currency formatting parsed", b.max_loss_week == 1250.0, f"got {b.max_loss_week}")
check("garbage disables that limit rather than crashing",
      b.max_premium_per_trade == 0.0, f"got {b.max_premium_per_trade}")
del os.environ["RISK_MAX_LOSS_PER_WEEK"], os.environ["RISK_MAX_PREMIUM_PER_TRADE"]

print("\n12. /budget output")
b = RiskBudget(db_path=db_with(("2026-09-02", -250.0)),
               max_loss_week=500, max_loss_month=1500, max_premium_per_trade=400,
               max_open_positions=3)
out = b.describe(TODAY)
check("shows usage against the weekly limit", "$-250.00" in out and "500.00" in out)
check("says entries are active while inside budget", "Within budget" in out)
b2 = RiskBudget(db_path=db_with(("2026-09-02", -900.0)),
                max_loss_week=500, max_loss_month=0, max_premium_per_trade=0,
                max_open_positions=0)
out2 = b2.describe(TODAY)
check("says HALTED when breached", "HALTED" in out2)
check("states that exit warnings continue", "Exit warnings" in out2)

print("\n" + "=" * 62)
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
