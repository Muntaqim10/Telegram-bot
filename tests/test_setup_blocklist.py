"""Retiring a setup is config, not a code change.

The per-setup table looks decisive -- the highest-volume setup shows 81 alerts at 45.7%
and -3.29% over three days -- but every setup has fired on only 3-6 distinct days, and
alerts within a day move together. Tested on the day count rather than the alert count,
no setup differs from a coin flip. So the blocklist ships empty: the switch exists, and
scripts/check_setup_performance.py decides when to pull it.
"""
import importlib

import pytest

import src.execution.signal_pipeline as sp


def reload_with(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SETUP_BLOCKLIST", raising=False)
    else:
        monkeypatch.setenv("SETUP_BLOCKLIST", value)
    return importlib.reload(sp)


@pytest.fixture(autouse=True)
def restore():
    yield
    import os
    os.environ.pop("SETUP_BLOCKLIST", None)
    importlib.reload(sp)


class TestTheDefault:
    def test_nothing_is_blocked_out_of_the_box(self, monkeypatch):
        """Five days of data does not justify retiring a setup."""
        assert reload_with(monkeypatch, None).SETUP_BLOCKLIST == frozenset()

    def test_an_empty_value_blocks_nothing(self, monkeypatch):
        assert reload_with(monkeypatch, "").SETUP_BLOCKLIST == frozenset()

    def test_a_list_of_commas_blocks_nothing(self, monkeypatch):
        assert reload_with(monkeypatch, " , ,  ").SETUP_BLOCKLIST == frozenset()


class TestParsing:
    def test_one_setup(self, monkeypatch):
        m = reload_with(monkeypatch, "Closing Range Put Breakdown (CRB)")
        assert "Closing Range Put Breakdown (CRB)" in m.SETUP_BLOCKLIST

    def test_several_with_padding(self, monkeypatch):
        m = reload_with(monkeypatch, " Trend Breakout , Volume Breakdown ")
        assert m.SETUP_BLOCKLIST == frozenset({"Trend Breakout", "Volume Breakdown"})

    def test_emoji_names_survive(self, monkeypatch):
        """Live catalyst_type strings carry emoji; they are the join key."""
        name = "🔥 DUAL ORB+CRB CALL BREAKOUT"
        assert name in reload_with(monkeypatch, name).SETUP_BLOCKLIST

    def test_matching_is_exact_not_substring(self, monkeypatch):
        """'CRB' must not silently retire four different setups."""
        m = reload_with(monkeypatch, "CRB")
        assert "Closing Range Put Breakdown (CRB)" not in m.SETUP_BLOCKLIST


class TestTheGate:
    def test_it_runs_before_the_position_is_registered(self):
        """A gate after add_position() must remove_position() first. This one sits above
        it, so there is nothing to clean up -- and that ordering is the reason."""
        import inspect

        src = inspect.getsource(sp)
        gate = src.index("BLOCKED by setup blocklist")
        add = src.index("risk_manager.add_position")
        assert gate < add, "the blocklist gate must precede the speculative add_position"

    def test_the_gate_reads_catalyst_type(self):
        import inspect

        src = inspect.getsource(sp.SignalPipeline.process_signal)
        assert 'signal.get("catalyst_type")' in src
