"""Manual test for the Robinhood statement parser.

The statement is the only ground truth this system has, so a parsing error here
silently corrupts every downstream number. These cases cover the formats that
actually appear in a Robinhood activity export.

Run: python scripts/manual_test_statement_import.py
"""
import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_spec = importlib.util.spec_from_file_location(
    "rh_import", os.path.join(os.path.dirname(__file__), "import_robinhood.py"))
rh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rh)

failures = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


HEADER = ('"Activity Date","Process Date","Settle Date","Instrument","Description",'
          '"Trans Code","Quantity","Price","Amount"\n')


def statement(*rows):
    path = os.path.join(tempfile.mkdtemp(), "statement.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER)
        for r in rows:
            f.write(r + "\n")
    return path


def row(date, ticker, desc, code, qty="1", price="", amount=""):
    return f'"{date}","{date}","{date}","{ticker}","{desc}","{code}","{qty}","{price}","{amount}"'


print("\n1. Money parsing -- debits are written in parentheses")
check("($525.04) is negative", rh.money("($525.04)") == -525.04)
check("$534.93 is positive", rh.money("$534.93") == 534.93)
check("thousands separators", rh.money("($1,035.65)") == -1035.65)
check("empty is zero", rh.money("") == 0.0)
check("garbage is zero, not a crash", rh.money("n/a") == 0.0)

print("\n2. Contract descriptions")
check("plain contract", rh.contract_key("AMZN 8/31/2026 Call $260.00") ==
      ("AMZN", "8/31/2026", "Call", "260.00"))
check("expiration prefix is tolerated",
      rh.contract_key("Option Expiration for NVDA 8/21/2026 Call $250.00") ==
      ("NVDA", "8/21/2026", "Call", "250.00"))
check("strike with thousands separator",
      rh.contract_key("SPXW 5/13/2026 Call $5,890.00")[3] == "5890.00")
check("non-option rows are ignored", rh.contract_key("ACH Deposit") is None)

print("\n3. A bought-then-sold contract")
p = statement(
    row("08/25/2026", "PLTR", "PLTR 9/18/2026 Call $150.00", "BTO", "1", "$5.00", "($500.04)"),
    row("08/27/2026", "PLTR", "PLTR 9/18/2026 Call $150.00", "STC", "1", "$8.00", "$799.90"),
)
f = rh.build_fills(rh.parse_statement(p))
check("one contract built", len(f) == 1, f"got {len(f)}")
check("exit kind SOLD", f[0]["exit_kind"] == "SOLD")
check("P&L is proceeds minus cost", abs(f[0]["pnl"] - 299.86) < 0.01, f"got {f[0]['pnl']}")
check("hold length in days", f[0]["days_held"] == 2)
check("expiration normalised to ISO", f[0]["expiration"] == "2026-09-18",
      f"got {f[0]['expiration']}")

print("\n4. A contract left to expire worthless")
p = statement(
    row("08/19/2026", "NVDA", "NVDA 8/21/2026 Call $230.00", "BTO", "1", "$0.14", "($14.04)"),
    row("08/21/2026", "NVDA", "Option Expiration for NVDA 8/21/2026 Call $230.00", "OEXP", "1"),
)
f = rh.build_fills(rh.parse_statement(p))
check("exit kind EXPIRED", f[0]["exit_kind"] == "EXPIRED")
check("total loss of premium", abs(f[0]["pnl"] - (-14.04)) < 0.01, f"got {f[0]['pnl']}")
check("-100% return", abs(f[0]["pnl_pct"] - (-1.0)) < 1e-9, f"got {f[0]['pnl_pct']}")

print("\n5. A SHORT leg expiring is not the trader's loss")
p = statement(
    row("08/19/2026", "NVDA", "NVDA 8/21/2026 Call $250.00", "STO", "1S", "$0.01", "$0.94"),
    row("08/21/2026", "NVDA", "Option Expiration for NVDA 8/21/2026 Call $250.00", "OEXP", "1S"),
)
f = rh.build_fills(rh.parse_statement(p))
check("short premium is not treated as a bought contract", len(f) == 0, f"got {len(f)}")

print("\n6. An exercised contract is not a wipeout")
p = statement(
    row("09/12/2026", "SOFI", "SOFI 9/19/2026 Call $26.00", "BTO", "57", "$1.07", "($6,134.38)"),
    row("09/19/2026", "SOFI", "SOFI 9/19/2026 Call $26.00", "OEXCS", "57"),
)
f = rh.build_fills(rh.parse_statement(p))
check("exit kind EXERCISED", f[0]["exit_kind"] == "EXERCISED")
check("no P&L invented for an exercise", f[0]["pnl"] is None)
check("cost still recorded", abs(f[0]["cost"] - 6134.38) < 0.01)

print("\n7. A still-open contract")
p = statement(
    row("08/28/2026", "CVX", "CVX 12/18/2026 Call $230.00", "BTO", "1", "$6.41", "($641.04)"),
)
f = rh.build_fills(rh.parse_statement(p))
check("exit kind OPEN", f[0]["exit_kind"] == "OPEN")
check("no close date", f[0]["close_date"] is None)
check("excluded from P&L stats", f[0]["pnl_pct"] is None)

print("\n8. Scaling into one contract across two fills")
p = statement(
    row("08/25/2026", "AMD", "AMD 9/18/2026 Call $180.00", "BTO", "1", "$4.00", "($400.04)"),
    row("08/26/2026", "AMD", "AMD 9/18/2026 Call $180.00", "BTO", "1", "$3.00", "($300.04)"),
    row("08/28/2026", "AMD", "AMD 9/18/2026 Call $180.00", "STC", "2", "$5.00", "$999.80"),
)
f = rh.build_fills(rh.parse_statement(p))
check("both entries rolled into one contract", len(f) == 1 and f[0]["quantity"] == 2,
      f"got {len(f)} fills, qty {f[0]['quantity'] if f else '-'}")
check("cost is the sum of both fills", abs(f[0]["cost"] - 700.08) < 0.01, f"got {f[0]['cost']}")
check("open date is the FIRST entry", f[0]["open_date"] == "2026-08-25")

print("\n9. Matching fills to alerts")
fills = [
    {"ticker": "PLTR", "open_date": "2026-08-25", "matched_alert_id": None, "alert_lag_days": None},
    {"ticker": "ZZZZ", "open_date": "2026-08-25", "matched_alert_id": None, "alert_lag_days": None},
    {"ticker": "PLTR", "open_date": "2026-08-01", "matched_alert_id": None, "alert_lag_days": None},
]
import datetime
alerts = [{"id": 7, "ticker": "PLTR", "_date": datetime.date(2026, 8, 24)}]
n = rh.match(fills, alerts, window_days=3)
check("entry 1 day after the alert matches", fills[0]["matched_alert_id"] == 7)
check("lag recorded", fills[0]["alert_lag_days"] == 1)
check("unalerted ticker stays unmatched", fills[1]["matched_alert_id"] is None)
check("entry BEFORE the alert does not match", fills[2]["matched_alert_id"] is None)
check("match count", n == 1, f"got {n}")

print("\n10. Import is idempotent (safe to re-run on overlapping statements)")
import src.database as db
db._init_db_sync()
sample = [{
    "contract": "TEST 2026-09-18 Call 1.0", "ticker": "TEST", "option_type": "CALL",
    "strike": 1.0, "expiration": "2026-09-18", "open_date": "2026-08-25",
    "close_date": "2026-08-27", "quantity": 1, "entry_price": 5.0, "exit_price": 8.0,
    "cost": 500.0, "proceeds": 800.0, "pnl": 300.0, "pnl_pct": 0.6, "days_held": 2,
    "exit_kind": "SOLD", "matched_alert_id": None, "alert_lag_days": None,
}]
db._upsert_real_fills_sync(sample)
before = len([r for r in db.get_real_fills_sync() if r["ticker"] == "TEST"])
db._upsert_real_fills_sync(sample)
after = [r for r in db.get_real_fills_sync() if r["ticker"] == "TEST"]
check("re-import does not duplicate", before == 1 and len(after) == 1,
      f"before {before}, after {len(after)}")
sample[0]["pnl"] = 999.0
db._upsert_real_fills_sync(sample)
updated = [r for r in db.get_real_fills_sync() if r["ticker"] == "TEST"]
check("re-import updates in place", updated[0]["pnl"] == 999.0, f"got {updated[0]['pnl']}")

# clean up the synthetic row
import sqlite3
_c = sqlite3.connect(db.DB_PATH)
_c.execute("DELETE FROM real_fills WHERE ticker = 'TEST'")
_c.commit()
_c.close()

print("\n" + "=" * 60)
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
