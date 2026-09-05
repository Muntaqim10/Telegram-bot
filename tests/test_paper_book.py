"""Forward paper trading: the only way to grade an alerter nobody trades.

523 alerts, zero confirmed positions. Confirmation cannot close that gap when the trader
does not take the alerts, and the backtest cannot either -- it prices with Black-Scholes
on realized volatility and tests four setups that share nothing with the live stream.

Paper trading forward can, because the chain is retrievable in real time: each alert
opens a position on the contract it actually named, at the quoted ask, and closes at the
real bid. These tests pin the parts that decide whether the resulting numbers are honest
or flattering -- the spread being paid twice, and an unquotable contract costing money
rather than quietly disappearing from the sample.
"""
import asyncio
import sqlite3

import pytest

import src.database as db
from src.execution import paper_book


@pytest.fixture
def paper_db(tmp_path, monkeypatch):
    """A throwaway database with the real paper_trades schema."""
    path = str(tmp_path / "paper.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER, alert_table TEXT,
        opened_at TEXT NOT NULL, ticker TEXT NOT NULL, setup TEXT, direction TEXT,
        occ_symbol TEXT NOT NULL, strike REAL, expiry TEXT, option_type TEXT,
        entry_ask REAL NOT NULL, entry_bid REAL, entry_spot REAL, entry_delta REAL,
        entry_theta REAL, entry_iv REAL, marked_at TEXT, mark_bid REAL,
        closed_at TEXT, exit_bid REAL, exit_reason TEXT, pnl_pct REAL, days_held INTEGER,
        UNIQUE(occ_symbol, opened_at))""")
    conn.commit()
    conn.close()
    return path


def open_trade(**kw):
    row = {"opened_at": "2026-09-01 10:00:00", "ticker": "NVDA", "setup": "TEST",
           "direction": "Long", "occ_symbol": "NVDA260918C00230000",
           "expiry": "2026-09-18", "entry_ask": 10.00, "entry_bid": 9.80}
    row.update(kw)
    return asyncio.run(db.open_paper_trade(row))


def rows(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    out = [dict(r) for r in conn.execute("SELECT * FROM paper_trades")]
    conn.close()
    return out


def sweep_with(quotes, monkeypatch):
    async def fake(symbols, session=None, token=None):
        return quotes
    monkeypatch.setattr(paper_book, "fetch_option_quotes", fake)
    return asyncio.run(paper_book.sweep())


class TestOpening:
    def test_an_alert_opens_a_paper_trade(self, paper_db):
        assert open_trade() is True
        assert len(rows(paper_db)) == 1

    def test_entry_is_the_ask_not_the_mid(self, paper_db):
        """A taker pays the ask. The old backtest used a theoretical mid, which
        flatters every result by roughly half the spread."""
        open_trade(entry_ask=10.00, entry_bid=9.80)
        assert rows(paper_db)[0]["entry_ask"] == 10.00

    def test_no_contract_means_no_paper_trade(self, paper_db):
        assert open_trade(occ_symbol=None) is False
        assert rows(paper_db) == []

    def test_no_ask_means_no_paper_trade(self, paper_db):
        """Nothing to buy at, so there is nothing to grade."""
        assert open_trade(entry_ask=None) is False

    def test_the_same_contract_twice_in_one_second_is_one_trade(self, paper_db):
        assert open_trade() is True
        assert open_trade() is False
        assert len(rows(paper_db)) == 1


class TestMarking:
    def test_an_open_trade_is_marked_not_closed(self, paper_db, monkeypatch):
        open_trade(opened_at=paper_book._market_today().strftime("%Y-%m-%d 10:00:00"))
        r = sweep_with({"NVDA260918C00230000": {"bid": 11.0, "ask": 11.2}}, monkeypatch)
        assert r["marked"] == 1 and r["closed"] == 0
        assert rows(paper_db)[0]["mark_bid"] == 11.0
        assert rows(paper_db)[0]["closed_at"] is None


class TestClosing:
    def test_the_hold_ceiling_closes_it_at_the_bid(self, paper_db, monkeypatch):
        open_trade()  # opened 2026-09-01, long past the ceiling
        r = sweep_with({"NVDA260918C00230000": {"bid": 12.0, "ask": 12.4}}, monkeypatch)
        assert r["closed"] == 1
        row = rows(paper_db)[0]
        assert row["exit_bid"] == 12.0, "exit is the bid, not the ask or the mid"
        assert row["exit_reason"] == "HOLD_CEILING"

    def test_pnl_is_ask_to_bid_so_the_spread_is_paid_twice(self, paper_db, monkeypatch):
        open_trade(entry_ask=10.00)
        sweep_with({"NVDA260918C00230000": {"bid": 11.0, "ask": 11.2}}, monkeypatch)
        # bought at 10.00, sold at 11.00 -> +10%, not the +12% a mid-to-mid mark implies
        assert rows(paper_db)[0]["pnl_pct"] == pytest.approx(10.0)

    def test_an_unquotable_contract_settles_at_zero(self, paper_db, monkeypatch):
        """Illiquidity is a real cost. Skipping these would quietly select for the
        contracts that happened to stay liquid."""
        open_trade(entry_ask=4.00)
        r = sweep_with({}, monkeypatch)
        assert r["closed"] == 1
        row = rows(paper_db)[0]
        assert row["exit_bid"] == 0.0
        assert row["pnl_pct"] == pytest.approx(-100.0)

    def test_expiry_closes_it_even_inside_the_hold_window(self, paper_db, monkeypatch):
        today = paper_book._market_today()
        open_trade(opened_at=today.strftime("%Y-%m-%d 10:00:00"),
                   expiry=today.strftime("%Y-%m-%d"))
        r = sweep_with({"NVDA260918C00230000": {"bid": 0.4, "ask": 0.5}}, monkeypatch)
        assert rows(paper_db)[0]["exit_reason"] == "EXPIRED"
        assert r["closed"] == 1

    def test_a_closed_trade_is_not_swept_again(self, paper_db, monkeypatch):
        open_trade()
        sweep_with({"NVDA260918C00230000": {"bid": 12.0, "ask": 12.4}}, monkeypatch)
        again = sweep_with({"NVDA260918C00230000": {"bid": 99.0, "ask": 99.5}}, monkeypatch)
        assert again == {"open": 0, "marked": 0, "closed": 0}
        assert rows(paper_db)[0]["exit_bid"] == 12.0


class TestItNeverBreaksTheAlert:
    def test_an_empty_book_is_a_no_op(self, paper_db, monkeypatch):
        assert sweep_with({}, monkeypatch) == {"open": 0, "marked": 0, "closed": 0}
