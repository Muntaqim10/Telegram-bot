"""Expiration warnings: the guard against contracts left to expire worthless.

24 contracts in the account's history expired for $6,008 with no exit decision made.
"""
import pytest

from src.execution.risk_manager import RiskManager
from tests.conftest import EXPIRED_YESTERDAY, EXPIRES_IN_A_WEEK, EXPIRES_TOMORROW, TODAY


@pytest.fixture
def book(rm, add):
    """One position expiring tomorrow, one in a week, one priced via the live path."""
    add(rm, "AAPL", entry_price=230.0, initial_atr=2.0, direction="Long",
        option_expiration=EXPIRES_TOMORROW)
    add(rm, "MSFT", entry_price=500.0, initial_atr=3.0, direction="Long",
        option_expiration=EXPIRES_IN_A_WEEK)
    add(rm, "NVDA", entry_price=180.0, initial_atr=4.0, direction="Long")
    return rm


def test_add_position_stores_the_expiration(rm, add):
    add(rm, "AAPL", entry_price=230.0, initial_atr=2.0, direction="Long",
        option_expiration=EXPIRES_TOMORROW)
    assert rm.active_positions["AAPL"]["option_expiration"] == EXPIRES_TOMORROW
    assert rm.active_positions["AAPL"]["expiration_warned"] is False


def test_expiration_arrives_via_the_live_pricing_path(rm, add):
    """The pipeline registers a position before it resolves the option chain, so the
    expiration lands in attach_option_pricing rather than add_position."""
    add(rm, "NVDA", entry_price=180.0, initial_atr=4.0, direction="Long")
    assert rm.active_positions["NVDA"]["option_expiration"] is None

    rm.attach_option_pricing("NVDA", option_entry_price=8.10, option_entry_delta=0.75,
                             option_entry_theta=-0.12, option_expiration=EXPIRES_TOMORROW)
    assert rm.active_positions["NVDA"]["option_expiration"] == EXPIRES_TOMORROW


def test_warns_only_inside_the_two_day_window(book):
    book.attach_option_pricing("NVDA", option_entry_price=8.10, option_entry_delta=0.75,
                               option_entry_theta=-0.12, option_expiration=EXPIRES_TOMORROW)
    warnings = book.check_expiration_warnings(TODAY)

    assert sorted(w["ticker"] for w in warnings) == ["AAPL", "NVDA"]
    assert all(w["days_to_expiration"] == 1 for w in warnings)
    assert book.active_positions["AAPL"]["expiration_warned"] is True
    assert book.active_positions["MSFT"]["expiration_warned"] is False, \
        "a position 7 days out must stay quiet"


def test_does_not_warn_twice_on_the_same_day(book):
    book.check_expiration_warnings(TODAY)
    assert book.check_expiration_warnings(TODAY) == []


def test_position_without_an_expiration_is_skipped(rm, add):
    add(rm, "SPY", entry_price=660.0, initial_atr=3.0, direction="Long")
    assert rm.check_expiration_warnings(TODAY) == []


class TestDailyReset:
    def test_carries_live_options_and_drops_the_rest(self, rm, add):
        add(rm, "TSLA", entry_price=400.0, initial_atr=5.0, direction="Long",
            option_expiration=EXPIRED_YESTERDAY)
        add(rm, "AMD", entry_price=160.0, initial_atr=2.0, direction="Long",
            option_expiration=EXPIRES_TOMORROW)
        add(rm, "F", entry_price=12.0, initial_atr=0.3, direction="Long")  # intraday
        rm.active_positions["AMD"]["expiration_warned"] = True

        rm.reset_daily(current_date=TODAY)

        assert sorted(rm.active_positions) == ["AMD"]
        assert rm.active_positions["AMD"]["expiration_warned"] is False, \
            "a carried position must be able to warn again today"
        assert [w["ticker"] for w in rm.check_expiration_warnings(TODAY)] == ["AMD"]

    def test_without_a_date_clears_everything(self, rm, add):
        add(rm, "AMD", entry_price=160.0, initial_atr=2.0, direction="Long",
            option_expiration=EXPIRES_TOMORROW)
        rm.reset_daily()
        assert rm.active_positions == {}

    def test_unparseable_expiration_is_kept_not_silently_dropped(self, rm, add):
        """Dropping it would abandon a real open contract -- the failure this feature
        exists to prevent, triggered by a formatting change."""
        add(rm, "CVX", entry_price=230.0, initial_atr=2.0, direction="Long",
            option_expiration="2026-09-18T00:00:00", entry_date="2026-08-26")
        rm.reset_daily(current_date=TODAY)
        assert "CVX" in rm.active_positions


def test_requeued_warning_fires_again(rm, add):
    """What main._dispatch_expiration_warnings does when a Telegram send fails."""
    add(rm, "META", entry_price=700.0, initial_atr=5.0, direction="Long",
        option_expiration=EXPIRES_TOMORROW)
    assert [w["ticker"] for w in rm.check_expiration_warnings(TODAY)] == ["META"]
    assert rm.check_expiration_warnings(TODAY) == []

    rm.requeue_warning("META", "expiration_warned")
    assert [w["ticker"] for w in rm.check_expiration_warnings(TODAY)] == ["META"]


class TestPersistence:
    def test_survives_a_restart(self, state_file, add):
        first = RiskManager(state_path=state_file)
        add(first, "GOOG", entry_price=250.0, initial_atr=3.0, direction="Long",
            option_expiration=EXPIRES_TOMORROW)
        first.attach_option_pricing("GOOG", option_entry_price=6.40, option_entry_delta=0.75,
                                    option_entry_theta=-0.09, option_expiration=EXPIRES_TOMORROW)
        import os
        assert os.path.exists(state_file)

        restarted = RiskManager(state_path=state_file)
        assert restarted.active_positions == {}
        assert restarted.load_positions(current_date=TODAY) == 1
        assert restarted.active_positions["GOOG"]["option_expiration"] == EXPIRES_TOMORROW
        assert restarted.active_positions["GOOG"]["entry_price"] == 250.0
        assert [w["ticker"] for w in restarted.check_expiration_warnings(TODAY)] == ["GOOG"]

    def test_expired_contract_does_not_come_back(self, state_file, add):
        seed = RiskManager(state_path=state_file)
        add(seed, "INTC", entry_price=25.0, initial_atr=0.5, direction="Long",
            option_expiration=EXPIRED_YESTERDAY)

        restarted = RiskManager(state_path=state_file)
        restarted.load_positions(current_date=TODAY)
        assert "INTC" not in restarted.active_positions

    def test_corrupt_state_file_degrades_to_empty(self, state_file):
        with open(state_file, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        assert RiskManager(state_path=state_file).load_positions(current_date=TODAY) == 0

    def test_load_refuses_to_clobber_live_positions(self, state_file, add):
        seed = RiskManager(state_path=state_file)
        add(seed, "GOOG", entry_price=250.0, initial_atr=3.0, direction="Long",
            option_expiration=EXPIRES_TOMORROW)

        live = RiskManager(state_path=state_file)
        add(live, "AMD", entry_price=160.0, initial_atr=2.0, direction="Long",
            option_expiration=EXPIRES_TOMORROW)
        assert live.load_positions(current_date=TODAY) == 0
        assert "AMD" in live.active_positions
