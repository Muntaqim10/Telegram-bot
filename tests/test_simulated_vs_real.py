"""Simulation must never be presented as a trade the account actually made.

The bot alerts; execution happens by hand. But add_position() is speculative -- every
alert registers a position whether or not the trader took it -- and the trailing stop
then runs on all of them. So the engine produces exits, outcomes and per-catalyst
statistics for contracts nobody ever bought: 249 "TRADE CLOSED" events and 873 outcome
rows against zero confirmed positions.

Nothing downstream restated that. The alert rendered a "Catalyst Reliability" line, the
outcome log looked like a trade history, and the statement importer reported "alert
follow-through". These tests hold the line between the two.
"""
import csv
import os

import pytest

from src.alerts import AlertGateway
from src.execution.risk_manager import OUTCOME_CSV_COLUMNS


@pytest.fixture
def outcome_csv(tmp_path, monkeypatch):
    """Redirect close_trade's outcome log into tmp_path.

    close_trade builds the path with os.path.abspath at call time, so intercepting that
    one call keeps the real data/trade_outcomes.csv untouched.
    """
    path = tmp_path / "trade_outcomes.csv"
    real_abspath = os.path.abspath

    def fake_abspath(p):
        return str(path) if "trade_outcomes" in str(p) else real_abspath(p)

    monkeypatch.setattr(os.path, "abspath", fake_abspath)
    return path


class TestTheOutcomeLogSaysWhichIsWhich:
    def test_an_unconfirmed_close_is_marked_simulated(self, rm, add, outcome_csv):
        add(rm, "SIM", entry_price=100.0, initial_atr=2.0, direction="Long",
            confirmed=False)
        rm.close_trade("SIM", "INVALIDATED", exit_price=98.0, pnl_pct=-0.02)
        assert list(csv.DictReader(outcome_csv.open()))[-1]["confirmed"] == "0"

    def test_a_confirmed_close_is_marked_real(self, rm, add, outcome_csv):
        add(rm, "REAL", entry_price=100.0, initial_atr=2.0, direction="Long",
            confirmed=True)
        rm.close_trade("REAL", "TP_HIT", exit_price=104.0, pnl_pct=0.04)
        assert list(csv.DictReader(outcome_csv.open()))[-1]["confirmed"] == "1"

    def test_the_column_exists_in_the_schema(self):
        assert "confirmed" in OUTCOME_CSV_COLUMNS, \
            "without it the training set cannot separate money from simulation"


class TestTheReliabilityLineIsLabelled:
    """Built from record_outcome(), which counts every simulated close."""

    @staticmethod
    def report():
        gw = AlertGateway.__new__(AlertGateway)
        gw.catalyst_stats = {
            "Breakout": {"total_signals": 9, "INVALIDATED": 4, "STOPPED_OUT": 2,
                         "TP_HIT": 3},
        }
        return gw.get_catalyst_report()

    def test_the_header_says_simulated(self):
        out = self.report()
        assert out is not None
        assert "simulated" in out.lower()

    def test_it_disclaims_trades_taken(self):
        assert "not trades taken" in self.report().lower()

    def test_the_per_row_count_is_not_called_trades(self):
        """'(9 trades)' reads as a track record; these were never traded."""
        assert "trades)" not in self.report()

    def test_a_thin_sample_still_says_nothing(self):
        gw = AlertGateway.__new__(AlertGateway)
        gw.catalyst_stats = {"Breakout": {"total_signals": 2, "INVALIDATED": 1,
                                          "STOPPED_OUT": 0, "TP_HIT": 1}}
        assert gw.get_catalyst_report() is None
