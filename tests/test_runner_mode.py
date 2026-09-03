"""Runner mode: what happens when the stock keeps going past the target.

A fixed take-profit closes at roughly 1.7x daily ATR while the downside stays open to
the whole premium, which truncates the right tail long options exist to capture.
"""
import pytest

from src.execution.risk_manager import RiskManager
from tests.conftest import TODAY

ATR = 2.0
EXPIRY = "2026-12-18"


@pytest.fixture
def runner():
    """A confirmed position with a $110 target on a $100 entry."""
    def _make(runner_mode, direction="Long", entry=100.0, tp=110.0):
        rm = RiskManager(state_path=None, runner_mode=runner_mode, runner_trail_atr=1.5)
        rm.add_position("SOAR", entry_price=entry, initial_atr=2.0, direction=direction,
                        stop_loss=(entry - 5 if direction == "Long" else entry + 5),
                        take_profit=tp, option_expiration=EXPIRY, confirmed=True)
        rm.attach_option_pricing("SOAR", option_entry_price=2.00, option_entry_delta=0.40,
                                 option_entry_theta=-0.05, option_expiration=EXPIRY)
        return rm
    return _make


def test_disabled_the_target_closes_the_position(runner):
    rm = runner(False)
    out = rm.update_trailing_stop("SOAR", current_price=112.0, current_atr=ATR,
                                  direction="Long")
    assert out["status"] == "TP_HIT"


class TestEnabled:
    def test_the_target_starts_a_trail_instead_of_closing(self, runner):
        rm = runner(True)
        out = rm.update_trailing_stop("SOAR", current_price=112.0, current_atr=ATR,
                                      direction="Long")

        assert out["status"] == "TP_HIT_TRAILING"
        assert "SOAR" in rm.active_positions
        assert rm.active_positions["SOAR"]["runner"] is True
        assert rm.active_positions["SOAR"]["trailing_stop"] == pytest.approx(109.0)
        assert "trailing_stop" in out, "the alert must carry the level"

    def test_a_soaring_stock_keeps_running(self, runner):
        rm = runner(True)
        rm.update_trailing_stop("SOAR", current_price=112.0, current_atr=ATR, direction="Long")
        for price in (120.0, 135.0, 150.0):
            out = rm.update_trailing_stop("SOAR", current_price=price, current_atr=ATR,
                                          direction="Long")
            assert out["status"] == "ACTIVE", f"closed early at ${price:.0f}"

        assert rm.active_positions["SOAR"]["trailing_stop"] == pytest.approx(147.0)
        assert rm.active_positions["SOAR"]["runner"] is True, "the target must not re-trigger"

    def test_the_trail_finally_closes_it_far_above_the_target(self, runner):
        rm = runner(True)
        for price in (112.0, 120.0, 135.0, 150.0):
            rm.update_trailing_stop("SOAR", current_price=price, current_atr=ATR,
                                    direction="Long")
        out = rm.update_trailing_stop("SOAR", current_price=146.0, current_atr=ATR,
                                      direction="Long")

        assert out["status"] == "RUNNER_STOPPED"
        assert out["exit_price"] == 146.0
        assert out["pnl"] == pytest.approx(0.46), "+46% rather than the +12% target"

    def test_the_trail_never_loosens(self, runner):
        rm = runner(True)
        rm.update_trailing_stop("SOAR", current_price=112.0, current_atr=ATR, direction="Long")
        rm.update_trailing_stop("SOAR", current_price=140.0, current_atr=ATR, direction="Long")
        high = rm.active_positions["SOAR"]["trailing_stop"]

        rm.update_trailing_stop("SOAR", current_price=138.0, current_atr=ATR, direction="Long")
        assert rm.active_positions["SOAR"]["trailing_stop"] == high

    def test_an_ordinary_stop_out_still_works(self, runner):
        rm = runner(True)
        out = rm.update_trailing_stop("SOAR", current_price=94.0, current_atr=ATR,
                                      direction="Long")
        assert out["status"] == "STOPPED_OUT"


class TestShortSide:
    def test_behaves_symmetrically(self, runner):
        rm = runner(True, direction="Short", entry=100.0, tp=90.0)

        out = rm.update_trailing_stop("SOAR", current_price=88.0, current_atr=ATR,
                                      direction="Short")
        assert out["status"] == "TP_HIT_TRAILING"
        assert rm.active_positions["SOAR"]["trailing_stop"] == pytest.approx(91.0)

        out = rm.update_trailing_stop("SOAR", current_price=70.0, current_atr=ATR,
                                      direction="Short")
        assert out["status"] == "ACTIVE"
        assert rm.active_positions["SOAR"]["trailing_stop"] == pytest.approx(73.0)

        out = rm.update_trailing_stop("SOAR", current_price=74.0, current_atr=ATR,
                                      direction="Short")
        assert out["status"] == "RUNNER_STOPPED"


def test_runner_state_survives_a_restart(state_file):
    rm = RiskManager(state_path=state_file, runner_mode=True, runner_trail_atr=1.5)
    rm.add_position("SOAR", entry_price=100.0, initial_atr=2.0, direction="Long",
                    stop_loss=95.0, take_profit=110.0, option_expiration=EXPIRY,
                    confirmed=True)
    rm.update_trailing_stop("SOAR", current_price=112.0, current_atr=ATR, direction="Long")

    restarted = RiskManager(state_path=state_file, runner_mode=True, runner_trail_atr=1.5)
    restarted.load_positions()
    assert restarted.active_positions["SOAR"]["runner"] is True
    assert restarted.active_positions["SOAR"]["trailing_stop"] == pytest.approx(109.0)
