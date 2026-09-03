"""Time stop and premium stop: the two exit failures visible in the account's history.

Hold past day 5 and the win rate falls from ~50% to 27%; 69 contracts were sold only
after losing 50-99% of premium.
"""
import pytest

from tests.conftest import TODAY

EXPIRY = "2026-12-18"


def reasons(alerts):
    return sorted(a["reason"] for a in alerts)


class TestTimeStop:
    def test_fires_past_the_hold_limit(self, rm, add):
        add(rm, "INTC", entry_price=40.0, initial_atr=1.0, direction="Long",
            option_expiration=EXPIRY, entry_date="2026-08-26")  # 7 days
        alerts = rm.check_position_health(TODAY)

        assert reasons(alerts) == ["TIME_STOP"]
        assert alerts[0]["days_held"] == 7
        assert rm.check_position_health(TODAY) == [], "must not re-fire the same day"

    def test_quiet_inside_the_window(self, rm, add):
        add(rm, "AAPL", entry_price=230.0, initial_atr=2.0, direction="Long",
            option_expiration=EXPIRY, entry_date="2026-08-31")  # 2 days
        assert rm.check_position_health(TODAY) == []

    def test_never_nags_someone_out_of_a_winner(self, rm, add):
        """The point of holding a long option is the move that keeps going."""
        add(rm, "WIN", entry_price=100.0, initial_atr=2.0, direction="Long",
            option_expiration=EXPIRY, entry_date="2026-08-20")  # 13 days
        rm.attach_option_pricing("WIN", option_entry_price=2.00, option_entry_delta=0.40,
                                 option_entry_theta=-0.01, option_expiration=EXPIRY)
        rm.update_trailing_stop("WIN", current_price=130.0, current_atr=2.0, direction="Long")
        assert "TIME_STOP" not in reasons(rm.check_position_health(TODAY))

    def test_still_fires_on_a_loser_held_too_long(self, rm, add):
        add(rm, "LOSE", entry_price=100.0, initial_atr=2.0, direction="Long",
            option_expiration=EXPIRY, entry_date="2026-08-20")
        rm.attach_option_pricing("LOSE", option_entry_price=2.00, option_entry_delta=0.40,
                                 option_entry_theta=-0.01, option_expiration=EXPIRY)
        rm.update_trailing_stop("LOSE", current_price=98.0, current_atr=2.0, direction="Long")
        assert "TIME_STOP" in reasons(rm.check_position_health(TODAY))


class TestPremiumStop:
    def test_fires_past_minus_fifty_percent(self, rm, add):
        add(rm, "GLD", entry_price=327.0, initial_atr=3.0, direction="Long",
            option_expiration=EXPIRY, entry_date="2026-09-01")
        rm.attach_option_pricing("GLD", option_entry_price=4.00, option_entry_delta=0.75,
                                 option_entry_theta=-0.10, option_expiration=EXPIRY)
        # -$3.00 on the underlying -> 4.00 - 2.25 delta - theta => about -56%
        rm.update_trailing_stop("GLD", current_price=324.0, current_atr=3.0, direction="Long")
        alerts = rm.check_position_health(TODAY)

        assert reasons(alerts) == ["PREMIUM_STOP"]
        assert alerts[0]["est_option_pnl_pct"] <= -0.50
        assert rm.check_position_health(TODAY) == []

    def test_quiet_on_a_modest_drawdown(self, rm, add):
        add(rm, "NVDA", entry_price=180.0, initial_atr=2.0, direction="Long",
            option_expiration=EXPIRY, entry_date="2026-09-01")
        rm.attach_option_pricing("NVDA", option_entry_price=8.00, option_entry_delta=0.75,
                                 option_entry_theta=-0.10, option_expiration=EXPIRY)
        rm.update_trailing_stop("NVDA", current_price=178.0, current_atr=2.0, direction="Long")
        assert rm.check_position_health(TODAY) == []

    def test_needs_option_data(self, rm, add):
        add(rm, "SPY", entry_price=660.0, initial_atr=3.0, direction="Long",
            entry_date="2026-09-01")
        rm.update_trailing_stop("SPY", current_price=600.0, current_atr=3.0, direction="Long")
        assert rm.check_position_health(TODAY) == []

    def test_values_a_put_the_right_way_round(self, rm, add):
        add(rm, "AMD", entry_price=165.0, initial_atr=2.0, direction="Short",
            option_expiration=EXPIRY, entry_date="2026-09-01")
        rm.attach_option_pricing("AMD", option_entry_price=4.00, option_entry_delta=-0.75,
                                 option_entry_theta=-0.10, option_expiration=EXPIRY)
        rm.update_trailing_stop("AMD", current_price=169.0, current_atr=2.0, direction="Short")
        assert reasons(rm.check_position_health(TODAY)) == ["PREMIUM_STOP"]

    def test_ignores_a_stale_price(self, rm, add):
        """A rotated-out ticker stops streaming; a stop fired on a days-old price is
        worse than no stop."""
        add(rm, "UBER", entry_price=77.0, initial_atr=1.0, direction="Long",
            option_expiration=EXPIRY, entry_date="2026-08-26")
        rm.attach_option_pricing("UBER", option_entry_price=4.00, option_entry_delta=0.75,
                                 option_entry_theta=-0.05, option_expiration=EXPIRY)
        rm.update_trailing_stop("UBER", current_price=70.0, current_atr=1.0, direction="Long")
        rm.active_positions["UBER"]["last_price_ts"] = 0  # simulate no recent tick

        alerts = rm.check_position_health(TODAY)
        assert "PREMIUM_STOP" not in reasons(alerts)
        assert reasons(alerts) == ["TIME_STOP"]
        assert alerts[0]["est_option_pnl_pct"] is None, \
            "an unvalued position must not report an estimate"


class TestThetaIsPricedIn:
    """A delta-only estimate is optimistic by exactly the amount that has decayed."""

    def _flat_position(self, rm, add, entry_date):
        add(rm, "MSFT", entry_price=500.0, initial_atr=3.0, direction="Long",
            option_expiration=EXPIRY, entry_date=entry_date)
        rm.attach_option_pricing("MSFT", option_entry_price=5.00, option_entry_delta=0.75,
                                 option_entry_theta=-0.30, option_expiration=EXPIRY)
        rm.update_trailing_stop("MSFT", current_price=500.0, current_atr=3.0, direction="Long")
        return rm

    def test_a_flat_stock_still_loses_to_theta(self, rm, add):
        self._flat_position(rm, add, "2026-08-26")  # 7 days
        est = rm._estimate_option_pnl(rm.active_positions["MSFT"], 500.0, TODAY)
        assert est < -0.35
        assert "PREMIUM_STOP" not in reasons(rm.check_position_health(TODAY)), \
            "-42% is correctly still short of the -50% line"

    def test_theta_alone_trips_the_stop_eventually(self, rm, add):
        self._flat_position(rm, add, "2026-08-23")  # 10 days
        assert "PREMIUM_STOP" in reasons(rm.check_position_health(TODAY))


def test_both_alerts_can_fire_for_one_position(rm, add):
    add(rm, "PLTR", entry_price=150.0, initial_atr=2.0, direction="Long",
        option_expiration=EXPIRY, entry_date="2026-08-20")
    rm.attach_option_pricing("PLTR", option_entry_price=5.00, option_entry_delta=0.75,
                             option_entry_theta=-0.10, option_expiration=EXPIRY)
    rm.update_trailing_stop("PLTR", current_price=145.0, current_atr=2.0, direction="Long")
    assert reasons(rm.check_position_health(TODAY)) == ["PREMIUM_STOP", "TIME_STOP"]


def test_reset_rearms_the_alerts(rm, add):
    add(rm, "INTC", entry_price=40.0, initial_atr=1.0, direction="Long",
        option_expiration=EXPIRY, entry_date="2026-08-26")
    rm.check_position_health(TODAY)
    assert rm.check_position_health(TODAY) == []

    rm.reset_daily(current_date=TODAY)
    assert "INTC" in rm.active_positions
    assert reasons(rm.check_position_health(TODAY)) == ["TIME_STOP"]


def test_requeue_lets_a_failed_dispatch_retry(rm, add):
    add(rm, "META", entry_price=700.0, initial_atr=5.0, direction="Long",
        option_expiration=EXPIRY, entry_date="2026-08-26")
    rm.check_position_health(TODAY)
    assert rm.check_position_health(TODAY) == []

    rm.requeue_warning("META", "time_stop_warned")
    assert reasons(rm.check_position_health(TODAY)) == ["TIME_STOP"]
