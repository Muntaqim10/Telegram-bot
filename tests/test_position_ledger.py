"""The alert/position split.

The bot cannot see Robinhood. An alert is a suggestion; a position exists only once the
trader confirms it with /took. Without this the bot nags about trades never made, and the
trader learns to ignore the alerts -- including the one that matters.
"""
import pytest

from src.alerts import AlertGateway
from src.execution.risk_manager import RiskManager
from tests.conftest import EXPIRES_TOMORROW, TODAY


@pytest.fixture
def alerted(make_rm):
    """A tracked alert: suggested by the bot, NOT confirmed by the trader."""
    def _alerted(ticker="PLTR", entry_date="2026-08-20", **kwargs):
        manager = make_rm()
        manager.add_position(ticker, entry_price=150.0, initial_atr=2.0, direction="Long",
                             option_expiration=EXPIRES_TOMORROW, entry_date=entry_date,
                             **kwargs)
        manager.attach_option_pricing(ticker, option_entry_price=5.00, option_entry_delta=0.75,
                                      option_entry_theta=-0.10,
                                      option_expiration=EXPIRES_TOMORROW)
        manager.update_trailing_stop(ticker, current_price=140.0, current_atr=2.0,
                                     direction="Long")
        return manager
    return _alerted


def test_unconfirmed_alert_produces_no_warnings(alerted):
    """13 days old, expiring tomorrow, premium collapsed -- and still silent."""
    manager = alerted()
    assert manager.active_positions["PLTR"]["confirmed"] is False
    assert manager.check_expiration_warnings(TODAY) == []
    assert manager.check_position_health(TODAY) == []


def test_confirming_turns_the_warnings_on(alerted):
    manager = alerted()
    manager.confirm_position("PLTR")
    assert manager.active_positions["PLTR"]["confirmed"] is True
    assert [w["ticker"] for w in manager.check_expiration_warnings(TODAY)] == ["PLTR"]


def test_the_real_fill_overrides_the_alert_time_quote(alerted):
    manager = alerted("AMD")
    manager.confirm_position("AMD", quantity=2, fill_price=6.25)
    pos = manager.active_positions["AMD"]

    assert pos["quantity"] == 2
    assert pos["fill_price"] == 6.25
    assert pos["option_entry_price"] == 6.25


def test_the_hold_clock_starts_at_confirmation(alerted):
    manager = alerted("NVDA", entry_date="2026-08-01")
    manager.confirm_position("NVDA", entry_date="2026-08-30")
    assert manager.days_held(manager.active_positions["NVDA"], TODAY) == 3


def test_daily_reset_drops_suggestions_and_keeps_holdings(rm):
    rm.add_position("SKIP", entry_price=10.0, initial_atr=1.0, direction="Long",
                    option_expiration=EXPIRES_TOMORROW)
    rm.add_position("HELD", entry_price=20.0, initial_atr=1.0, direction="Long",
                    option_expiration=EXPIRES_TOMORROW, confirmed=True)

    rm.reset_daily(current_date=TODAY)

    assert "SKIP" not in rm.active_positions
    assert "HELD" in rm.active_positions


class TestClosing:
    def test_records_the_real_option_pnl(self, alerted):
        manager = alerted("GLD")
        manager.confirm_position("GLD", quantity=1, fill_price=4.00)
        summary = manager.close_confirmed("GLD", exit_option_price=2.00)

        assert "GLD" not in manager.active_positions
        assert summary["option_pnl_pct"] == pytest.approx(-0.50)
        assert summary["days_held"] is not None

    def test_invents_nothing_without_a_fill_price(self, alerted):
        manager = alerted("UBER")
        manager.confirm_position("UBER")
        assert manager.close_confirmed("UBER")["option_pnl_pct"] is None


def test_list_positions_separates_holdings_from_suggestions(rm):
    rm.add_position("HOLD1", entry_price=10.0, initial_atr=1.0, direction="Long",
                    confirmed=True)
    rm.add_position("MAYBE", entry_price=20.0, initial_atr=1.0, direction="Long")
    book = rm.list_positions()

    assert [t for t, _ in book["confirmed"]] == ["HOLD1"]
    assert [t for t, _ in book["unconfirmed"]] == ["MAYBE"]


class TestTelegramCommands:
    @pytest.fixture
    def gateway(self, alerted):
        gw = AlertGateway(None)
        gw.attach_position_ledger(alerted("META"))
        return gw

    def test_took_records_quantity_and_fill(self, gateway):
        assert "Watching" in gateway._cmd_took("/took META 2 9.10")
        assert gateway.risk_manager.active_positions["META"]["fill_price"] == 9.10

    def test_took_explains_an_unknown_ticker(self, gateway):
        assert "No tracked alert" in gateway._cmd_took("/took ZZZZ")

    def test_took_tolerates_dollar_prefix_and_case(self, gateway):
        assert "No tracked alert" in gateway._cmd_took("/took $zzzz")

    def test_positions_lists_the_holding(self, gateway):
        gateway._cmd_took("/took META")
        assert "META" in gateway._cmd_positions()

    def test_closed_reports_a_result(self, gateway):
        gateway._cmd_took("/took META 1 9.10")
        assert "%" in gateway._cmd_closed("/closed META 8.96")

    def test_closed_handles_an_untracked_ticker(self, gateway):
        assert "not being tracked" in gateway._cmd_closed("/closed ZZZZ")

    def test_commands_degrade_without_a_ledger(self):
        assert "not attached" in AlertGateway(None)._cmd_took("/took META")
