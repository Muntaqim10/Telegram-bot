"""The circuit breaker: the one component whose job is to say no.

If it fails open it fails silently, and the trader finds out from their balance.
"""
import datetime
import os
import sqlite3

import pytest

from src.execution.risk_budget import RiskBudget

TODAY = datetime.date(2026, 9, 3)      # a Thursday; the week starts Monday 2026-08-31


@pytest.fixture
def budget(fills_db):
    """A RiskBudget over a throwaway database, with every limit off by default."""
    def _make(*fills, **limits):
        limits.setdefault("max_loss_week", 0)
        limits.setdefault("max_loss_month", 0)
        limits.setdefault("max_premium_per_trade", 0)
        limits.setdefault("max_open_positions", 0)
        return RiskBudget(db_path=fills_db(*fills), **limits)
    return _make


def test_with_no_limits_it_never_blocks(budget):
    b = budget(("2026-09-02", -5000.0))
    gate = b.check_entry(premium=99999.0, open_positions=99, today=TODAY)

    assert gate["allowed"] is True
    assert b.status(TODAY)["any_limit_set"] is False
    assert "No limits set" in b.describe(TODAY)


class TestLossCeilings:
    def test_weekly_ceiling_halts_entries(self, budget):
        b = budget(("2026-09-01", -300.0), ("2026-09-02", -250.0), max_loss_week=500)

        assert b.status(TODAY)["week_pnl"] == -550.0
        assert b.status(TODAY)["halted"] is True
        assert b.check_entry(today=TODAY)["allowed"] is False
        assert "weekly loss" in b.check_entry(today=TODAY)["reason"]

    def test_last_weeks_losses_do_not_count(self, budget):
        b = budget(("2026-08-20", -5000.0), max_loss_week=500)
        assert b.status(TODAY)["week_pnl"] == 0.0
        assert b.check_entry(today=TODAY)["allowed"] is True

    def test_monthly_ceiling_is_independent(self, budget):
        b = budget(("2026-09-01", -400.0), ("2026-09-02", -400.0), max_loss_month=700)
        assert b.status(TODAY)["month_pnl"] == -800.0
        assert b.check_entry(today=TODAY)["allowed"] is False
        assert "monthly loss" in b.check_entry(today=TODAY)["reason"]

    def test_wins_offset_losses_inside_the_period(self, budget):
        b = budget(("2026-09-01", -600.0), ("2026-09-02", 400.0), max_loss_week=500)
        assert b.status(TODAY)["week_pnl"] == -200.0
        assert b.check_entry(today=TODAY)["allowed"] is True

    def test_open_positions_are_not_realized_losses(self, budget, fills_db):
        path = fills_db(("2026-09-01", -900.0))
        conn = sqlite3.connect(path)
        conn.execute("INSERT INTO real_fills (contract, ticker, close_date, pnl) "
                     "VALUES (?,?,?,?)", ("OPEN", "Y", None, None))
        conn.commit()
        conn.close()

        b = RiskBudget(db_path=path, max_loss_week=1000, max_loss_month=0,
                       max_premium_per_trade=0, max_open_positions=0)
        assert b.status(TODAY)["week_pnl"] == -900.0


class TestPerTradeCaps:
    def test_premium_cap(self, budget):
        b = budget(max_premium_per_trade=500)
        assert b.check_entry(premium=400.0, today=TODAY)["allowed"] is True

        blocked = b.check_entry(premium=3350.0, today=TODAY)
        assert blocked["allowed"] is False
        assert "3,350" in blocked["reason"]

    def test_concurrency_cap(self, budget):
        b = budget(max_open_positions=3)
        assert b.check_entry(open_positions=2, today=TODAY)["allowed"] is True
        assert b.check_entry(open_positions=3, today=TODAY)["allowed"] is False


class TestFailsSafe:
    def test_a_missing_database_does_not_crash(self):
        b = RiskBudget(db_path="/nonexistent/path/to.db", max_loss_week=500,
                       max_loss_month=0, max_premium_per_trade=0, max_open_positions=0)
        assert b.status(TODAY)["week_pnl"] == 0.0

    def test_caps_needing_no_history_still_apply(self):
        b = RiskBudget(db_path="/nope.db", max_loss_week=0, max_loss_month=0,
                       max_premium_per_trade=100, max_open_positions=0)
        assert b.check_entry(premium=5000.0, today=TODAY)["allowed"] is False


class TestConfiguration:
    def test_currency_formatting_is_parsed(self, monkeypatch, fills_db):
        monkeypatch.setenv("RISK_MAX_LOSS_PER_WEEK", "$1,250")
        assert RiskBudget(db_path=fills_db()).max_loss_week == 1250.0

    def test_garbage_disables_that_limit(self, monkeypatch, fills_db):
        monkeypatch.setenv("RISK_MAX_PREMIUM_PER_TRADE", "not a number")
        assert RiskBudget(db_path=fills_db()).max_premium_per_trade == 0.0


class TestAnnouncement:
    def test_announces_once_per_day(self, budget):
        b = budget(("2026-09-02", -900.0), max_loss_week=500)
        assert b.should_announce_halt(TODAY) is True
        assert b.should_announce_halt(TODAY) is False
        assert b.should_announce_halt(TODAY + datetime.timedelta(days=1)) is True


class TestDescribe:
    def test_shows_usage_while_inside_budget(self, budget):
        out = budget(("2026-09-02", -250.0), max_loss_week=500, max_loss_month=1500,
                     max_premium_per_trade=400, max_open_positions=3).describe(TODAY)
        assert "$-250.00" in out and "500.00" in out
        assert "Within budget" in out

    def test_says_halted_and_that_exits_continue(self, budget):
        out = budget(("2026-09-02", -900.0), max_loss_week=500).describe(TODAY)
        assert "HALTED" in out
        assert "Exit warnings" in out
