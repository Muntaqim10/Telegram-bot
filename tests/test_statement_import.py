"""The Robinhood statement parser.

The statement is the only ground truth this system has, so a parsing error here silently
corrupts every downstream number.
"""
import datetime
import importlib.util
import os
import sqlite3

import pytest

_spec = importlib.util.spec_from_file_location(
    "rh_import",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "import_robinhood.py"))
rh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rh)

HEADER = ('"Activity Date","Process Date","Settle Date","Instrument","Description",'
          '"Trans Code","Quantity","Price","Amount"\n')


@pytest.fixture
def statement(tmp_path):
    def _write(*rows):
        path = tmp_path / "statement.csv"
        path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
        return str(path)
    return _write


def row(date, ticker, desc, code, qty="1", price="", amount=""):
    return f'"{date}","{date}","{date}","{ticker}","{desc}","{code}","{qty}","{price}","{amount}"'


@pytest.mark.parametrize("raw,expected", [
    ("($525.04)", -525.04),
    ("$534.93", 534.93),
    ("($1,035.65)", -1035.65),
    ("", 0.0),
    ("n/a", 0.0),
])
def test_money_parsing(raw, expected):
    """Robinhood writes debits in parentheses."""
    assert rh.money(raw) == expected


@pytest.mark.parametrize("desc,expected", [
    ("AMZN 8/31/2026 Call $260.00", ("AMZN", "8/31/2026", "Call", "260.00")),
    ("Option Expiration for NVDA 8/21/2026 Call $250.00",
     ("NVDA", "8/21/2026", "Call", "250.00")),
    ("ACH Deposit", None),
])
def test_contract_descriptions(desc, expected):
    assert rh.contract_key(desc) == expected


def test_strike_with_a_thousands_separator():
    assert rh.contract_key("SPXW 5/13/2026 Call $5,890.00")[3] == "5890.00"


class TestExitKinds:
    def test_bought_then_sold(self, statement):
        p = statement(
            row("08/25/2026", "PLTR", "PLTR 9/18/2026 Call $150.00", "BTO", "1", "$5.00", "($500.04)"),
            row("08/27/2026", "PLTR", "PLTR 9/18/2026 Call $150.00", "STC", "1", "$8.00", "$799.90"),
        )
        f = rh.build_fills(rh.parse_statement(p))
        assert len(f) == 1
        assert f[0]["exit_kind"] == "SOLD"
        assert f[0]["pnl"] == pytest.approx(299.86, abs=0.01)
        assert f[0]["days_held"] == 2
        assert f[0]["expiration"] == "2026-09-18", "expiration normalised to ISO"

    def test_expired_worthless(self, statement):
        p = statement(
            row("08/19/2026", "NVDA", "NVDA 8/21/2026 Call $230.00", "BTO", "1", "$0.14", "($14.04)"),
            row("08/21/2026", "NVDA", "Option Expiration for NVDA 8/21/2026 Call $230.00", "OEXP", "1"),
        )
        f = rh.build_fills(rh.parse_statement(p))
        assert f[0]["exit_kind"] == "EXPIRED"
        assert f[0]["pnl"] == pytest.approx(-14.04, abs=0.01)
        assert f[0]["pnl_pct"] == pytest.approx(-1.0)

    def test_a_short_leg_expiring_is_not_a_loss(self, statement):
        """Quantity 1S marks a short leg -- the trader KEPT that premium."""
        p = statement(
            row("08/19/2026", "NVDA", "NVDA 8/21/2026 Call $250.00", "STO", "1S", "$0.01", "$0.94"),
            row("08/21/2026", "NVDA", "Option Expiration for NVDA 8/21/2026 Call $250.00", "OEXP", "1S"),
        )
        assert rh.build_fills(rh.parse_statement(p)) == []

    def test_exercised_is_not_a_wipeout(self, statement):
        """The value moved into shares; counting it as a loss is the error that
        overstated the original expiry figure."""
        p = statement(
            row("09/12/2026", "SOFI", "SOFI 9/19/2026 Call $26.00", "BTO", "57", "$1.07", "($6,134.38)"),
            row("09/19/2026", "SOFI", "SOFI 9/19/2026 Call $26.00", "OEXCS", "57"),
        )
        f = rh.build_fills(rh.parse_statement(p))
        assert f[0]["exit_kind"] == "EXERCISED"
        assert f[0]["pnl"] is None
        assert f[0]["cost"] == pytest.approx(6134.38, abs=0.01)

    def test_still_open(self, statement):
        p = statement(
            row("08/28/2026", "CVX", "CVX 12/18/2026 Call $230.00", "BTO", "1", "$6.41", "($641.04)"),
        )
        f = rh.build_fills(rh.parse_statement(p))
        assert f[0]["exit_kind"] == "OPEN"
        assert f[0]["close_date"] is None
        assert f[0]["pnl_pct"] is None


def test_scaling_into_one_contract(statement):
    p = statement(
        row("08/25/2026", "AMD", "AMD 9/18/2026 Call $180.00", "BTO", "1", "$4.00", "($400.04)"),
        row("08/26/2026", "AMD", "AMD 9/18/2026 Call $180.00", "BTO", "1", "$3.00", "($300.04)"),
        row("08/28/2026", "AMD", "AMD 9/18/2026 Call $180.00", "STC", "2", "$5.00", "$999.80"),
    )
    f = rh.build_fills(rh.parse_statement(p))
    assert len(f) == 1
    assert f[0]["quantity"] == 2
    assert f[0]["cost"] == pytest.approx(700.08, abs=0.01)
    assert f[0]["open_date"] == "2026-08-25", "dated from the FIRST entry"


class TestAlertMatching:
    def test_matches_within_the_window_only(self):
        fills = [
            {"ticker": "PLTR", "open_date": "2026-08-25", "matched_alert_id": None,
             "matched_alert_table": None, "alert_lag_days": None},
            {"ticker": "ZZZZ", "open_date": "2026-08-25", "matched_alert_id": None,
             "matched_alert_table": None, "alert_lag_days": None},
            {"ticker": "PLTR", "open_date": "2026-08-01", "matched_alert_id": None,
             "matched_alert_table": None, "alert_lag_days": None},
        ]
        alerts = [{"id": 7, "ticker": "PLTR", "_date": datetime.date(2026, 8, 24),
                   "_table": "trade_log_archive"}]

        assert rh.match(fills, alerts, window_days=3) == 1
        assert fills[0]["matched_alert_id"] == 7
        assert fills[0]["alert_lag_days"] == 1
        assert fills[1]["matched_alert_id"] is None, "unalerted ticker"
        assert fills[2]["matched_alert_id"] is None, "entry BEFORE the alert"

    def test_the_side_must_agree(self):
        """The first real import matched 8 contracts, 5 of them the opposite side from
        the alert: the bot said puts on PLTR, AMZN and AAPL and the account bought calls.
        Ticker and date alone are not evidence."""
        fills = [
            {"ticker": "PLTR", "option_type": "CALL", "open_date": "2026-08-25",
             "matched_alert_id": None, "matched_alert_table": None, "alert_lag_days": None},
        ]
        alerts = [{"id": 7, "ticker": "PLTR", "direction": "Short",
                   "_date": datetime.date(2026, 8, 24), "_table": "trade_log"}]
        assert rh.match(fills, alerts, window_days=3) == 0
        assert fills[0]["matched_alert_id"] is None

    def test_the_same_side_still_matches(self):
        fills = [
            {"ticker": "PLTR", "option_type": "CALL", "open_date": "2026-08-25",
             "matched_alert_id": None, "matched_alert_table": None, "alert_lag_days": None},
        ]
        alerts = [{"id": 7, "ticker": "PLTR", "direction": "Long",
                   "_date": datetime.date(2026, 8, 24), "_table": "trade_log"}]
        assert rh.match(fills, alerts, window_days=3) == 1

    def test_one_alert_cannot_claim_both_sides(self):
        """A single GLD alert was matched to a call AND a put on the same strike, the
        same day. Whatever caused that straddle, it was not one directional alert."""
        fills = [
            {"ticker": "GLD", "option_type": "CALL", "open_date": "2026-08-24",
             "matched_alert_id": None, "matched_alert_table": None, "alert_lag_days": None},
            {"ticker": "GLD", "option_type": "PUT", "open_date": "2026-08-24",
             "matched_alert_id": None, "matched_alert_table": None, "alert_lag_days": None},
        ]
        alerts = [{"id": 46, "ticker": "GLD", "direction": "Long",
                   "_date": datetime.date(2026, 8, 24), "_table": "trade_log_archive"}]
        assert rh.match(fills, alerts, window_days=3) == 1
        assert [f["matched_alert_id"] for f in fills] == [46, None]

    @pytest.mark.parametrize("direction,expected", [
        ("Long", "CALL"), ("long", "CALL"), ("Short", "PUT"), ("short", "PUT"),
        ("", None), (None, None),
    ])
    def test_implied_side(self, direction, expected):
        assert rh.implied_option_type({"direction": direction}) == expected

    def test_an_alert_with_no_direction_does_not_block_a_match(self):
        """Older rows predate the column; absent is unknown, not disagreement."""
        fills = [{"ticker": "PLTR", "option_type": "CALL", "open_date": "2026-08-25",
                  "matched_alert_id": None, "matched_alert_table": None,
                  "alert_lag_days": None}]
        alerts = [{"id": 7, "ticker": "PLTR", "_date": datetime.date(2026, 8, 24),
                   "_table": "trade_log"}]
        assert rh.match(fills, alerts, window_days=3) == 1

    def test_records_which_table_the_alert_came_from(self):
        """trade_log and trade_log_archive autoincrement separately; 61 ids collide."""
        fills = [{"ticker": "PLTR", "open_date": "2026-08-25", "matched_alert_id": None,
                  "matched_alert_table": None, "alert_lag_days": None}]
        alerts = [{"id": 7, "ticker": "PLTR", "_date": datetime.date(2026, 8, 24),
                   "_table": "trade_log"}]
        rh.match(fills, alerts, window_days=3)
        assert fills[0]["matched_alert_table"] == "trade_log"


class TestIdempotency:
    """Re-importing an overlapping statement must update, never duplicate."""

    @pytest.fixture
    def isolated_db(self, tmp_path, monkeypatch):
        import src.database as db
        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
        monkeypatch.setattr(db, "CSV_LIVE_PATH", str(tmp_path / "nope_live.csv"))
        monkeypatch.setattr(db, "CSV_ARCHIVE_PATH", str(tmp_path / "nope_archive.csv"))
        db._init_db_sync()
        return db

    def _sample(self, pnl=300.0):
        return [{
            "contract": "TEST 2026-09-18 Call 1.0", "ticker": "TEST", "option_type": "CALL",
            "strike": 1.0, "expiration": "2026-09-18", "open_date": "2026-08-25",
            "close_date": "2026-08-27", "quantity": 1, "entry_price": 5.0, "exit_price": 8.0,
            "cost": 500.0, "proceeds": 800.0, "pnl": pnl, "pnl_pct": 0.6, "days_held": 2,
            "exit_kind": "SOLD", "matched_alert_id": None, "matched_alert_table": None,
            "alert_lag_days": None,
        }]

    def test_reimport_does_not_duplicate(self, isolated_db):
        isolated_db._upsert_real_fills_sync(self._sample())
        isolated_db._upsert_real_fills_sync(self._sample())
        assert len(isolated_db.get_real_fills_sync()) == 1

    def test_reimport_updates_in_place(self, isolated_db):
        isolated_db._upsert_real_fills_sync(self._sample())
        isolated_db._upsert_real_fills_sync(self._sample(pnl=999.0))
        assert isolated_db.get_real_fills_sync()[0]["pnl"] == 999.0


class TestExpiredWorthless:
    """A long contract past its expiry, never sold and never exercised, is a -100% loss.

    The importer only marked EXPIRED when the statement carried a matching OEXP row.
    31 contracts had none -- TSLA calls from 2019, BAC from 2020 -- so they sat as OPEN
    and were excluded from P&L entirely. That is why the account read +$16,682 when it
    had spent $16,474 on contracts that went to zero; the true figure is +$207.
    """

    def _fills(self, statement, expiry, opened):
        return rh.build_fills(rh.parse_statement(statement(
            f'"{opened}","{opened}","{opened}","ZZ","ZZ {expiry} Call $10.00",'
            f'"BTO","1","1.00","($100.00)"')))

    def test_a_past_expiry_with_no_close_is_a_total_loss(self, statement):
        f = self._fills(statement, "1/17/2020", "1/02/2020")
        assert len(f) == 1
        assert f[0]["exit_kind"] == "EXPIRED_WORTHLESS"
        assert f[0]["pnl"] == -100.0
        assert f[0]["pnl_pct"] == -1.0
        assert f[0]["proceeds"] == 0.0

    def test_it_is_dated_to_the_expiry(self, statement):
        f = self._fills(statement, "1/17/2020", "1/02/2020")
        assert f[0]["close_date"] == "2020-01-17"
        assert f[0]["days_held"] == 15

    def test_a_future_expiry_stays_open(self, statement):
        """Genuinely open positions must not be settled at -100%."""
        f = self._fills(statement, "12/19/2099", "1/02/2020")
        assert f[0]["exit_kind"] == "OPEN"
        assert f[0]["pnl"] is None

    def test_a_sold_contract_is_untouched(self, statement):
        f = rh.build_fills(rh.parse_statement(statement(
            '"1/02/2020","1/02/2020","1/02/2020","ZZ","ZZ 1/17/2020 Call $10.00",'
            '"BTO","1","1.00","($100.00)"',
            '"1/10/2020","1/10/2020","1/10/2020","ZZ","ZZ 1/17/2020 Call $10.00",'
            '"STC","1","1.50","$150.00"')))
        assert f[0]["exit_kind"] == "SOLD"
        assert f[0]["pnl"] == 50.0

    def test_an_exercised_contract_is_still_excluded(self, statement):
        """Its value moved into shares; the statement alone cannot price it, so it must
        not be counted as a wipeout."""
        f = rh.build_fills(rh.parse_statement(statement(
            '"1/02/2020","1/02/2020","1/02/2020","ZZ","ZZ 1/17/2020 Call $10.00",'
            '"BTO","1","1.00","($100.00)"',
            '"1/17/2020","1/17/2020","1/17/2020","ZZ","ZZ 1/17/2020 Call $10.00",'
            '"OEXCS","1","","$0.00"')))
        assert f[0]["exit_kind"] == "EXERCISED"
        assert f[0]["pnl"] is None
