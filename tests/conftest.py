"""Shared fixtures for the RiskManager / alerting suites.

Two rules matter for isolation:

* Never let a test touch data/active_positions.json or data/rallyhunter.db. The bot
  persists real position state there, and a test run must not be able to destroy it.
  `rm` and `db_path` exist so no test constructs a RiskManager with the default path.
* Only confirmed positions receive alerts. `add` therefore confirms by default, since
  most suites exercise the alerting rather than the confirmation gate itself.
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.risk_manager import RiskManager

TODAY = "2026-09-02"
EXPIRES_TOMORROW = "2026-09-03"
EXPIRES_IN_A_WEEK = "2026-09-09"
EXPIRED_YESTERDAY = "2026-09-01"


@pytest.fixture
def rm():
    """A RiskManager with persistence disabled."""
    return RiskManager(state_path=None)


@pytest.fixture
def make_rm():
    """Factory for suites needing several managers, or non-default thresholds."""
    def _make(**kwargs):
        kwargs.setdefault("state_path", None)
        return RiskManager(**kwargs)
    return _make


@pytest.fixture
def state_file(tmp_path):
    """A throwaway path for persistence round-trip tests."""
    return str(tmp_path / "active_positions.json")


@pytest.fixture
def add():
    """add_position, confirmed by default.

    The bot cannot see the brokerage account, so an unconfirmed alert is a suggestion
    rather than a holding and receives no warnings. Suites that exercise the gate itself
    pass confirmed=False explicitly.
    """
    def _add(manager, *args, **kwargs):
        kwargs.setdefault("confirmed", True)
        return manager.add_position(*args, **kwargs)
    return _add


@pytest.fixture
def db_path(tmp_path):
    """An empty database carrying only the tables under test."""
    path = str(tmp_path / "test.db")
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE real_fills (
        id INTEGER PRIMARY KEY AUTOINCREMENT, contract TEXT, ticker TEXT,
        close_date TEXT, pnl REAL)""")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def fills_db(db_path):
    """Seed the throwaway database with (close_date, pnl) closed contracts."""
    def _seed(*fills):
        conn = sqlite3.connect(db_path)
        conn.executemany(
            "INSERT INTO real_fills (contract, ticker, close_date, pnl) VALUES (?,?,?,?)",
            [(f"X {d}", "X", d, p) for d, p in fills])
        conn.commit()
        conn.close()
        return db_path
    return _seed
