"""The bot has to ask what was taken, or it never has a track record.

523 alerts, zero confirmed positions. Every performance figure the system reports --
catalyst reliability, calibration, per-setup win rates -- is the trailing stop's
simulation, and the statement importer cannot fill the gap: matching on ticker and date
paired five of eight fills to alerts pointing the opposite way.

/took has existed all along; nothing ever prompted for it. This prompt fires once after
the close, while the day's unconfirmed alerts still exist -- the daily reset drops them.
"""
import asyncio

import pytest

from src.execution.risk_manager import RiskManager


class Gateway:
    """Records what would have been sent, and can fail on demand."""

    def __init__(self, ok=True):
        self.sent = []
        self.ok = ok

    async def dispatch_informational(self, message):
        self.sent.append(message)
        return self.ok


@pytest.fixture
def engine(monkeypatch):
    """An IntradayEngine stub carrying only what the prompt touches."""
    import src.main as m

    eng = m.IntradayEngine.__new__(m.IntradayEngine)
    eng.risk_manager = RiskManager(state_path=None)
    eng.alerts = Gateway()
    return eng


def add(rm, ticker, confirmed, **kw):
    rm.add_position(ticker, kw.pop("entry_price", 100.0), initial_atr=2.0,
                    direction=kw.pop("direction", "Long"), confirmed=confirmed, **kw)


class TestWhatItSends:
    def test_it_lists_the_unconfirmed_alerts(self, engine):
        add(engine.risk_manager, "NVDA", False)
        add(engine.risk_manager, "AAPL", False)
        asyncio.run(engine._dispatch_confirmation_prompt(None))
        body = engine.alerts.sent[0]
        assert "/took NVDA" in body and "/took AAPL" in body

    def test_it_does_not_list_confirmed_holdings(self, engine):
        add(engine.risk_manager, "HELD", True)
        add(engine.risk_manager, "ALERT", False)
        asyncio.run(engine._dispatch_confirmation_prompt(None))
        body = engine.alerts.sent[0]
        assert "/took ALERT" in body
        assert "/took HELD" not in body

    def test_silence_when_there_is_nothing_to_confirm(self, engine):
        add(engine.risk_manager, "HELD", True)
        asyncio.run(engine._dispatch_confirmation_prompt(None))
        assert engine.alerts.sent == [], "an empty prompt is noise"

    def test_no_positions_at_all_sends_nothing(self, engine):
        asyncio.run(engine._dispatch_confirmation_prompt(None))
        assert engine.alerts.sent == []

    def test_it_says_what_happens_if_ignored(self, engine):
        """The cost of not answering is the whole reason to answer."""
        add(engine.risk_manager, "NVDA", False)
        asyncio.run(engine._dispatch_confirmation_prompt(None))
        assert "reset" in engine.alerts.sent[0].lower()

    def test_a_long_day_is_truncated_not_dumped(self, engine):
        """144 alerts landed on 2026-09-04. A 144-line message is unreadable, and
        Telegram is rate-limited per chat."""
        for i in range(40):
            add(engine.risk_manager, f"TK{i:02d}", False)
        asyncio.run(engine._dispatch_confirmation_prompt(None))
        body = engine.alerts.sent[0]
        # Count the listed tickers only. The header and the worked example both contain
        # "/took", so a naive substring count is two high.
        listed = [ln for ln in body.splitlines() if ln.strip().startswith("• <code>/took")]
        assert len(listed) == 25
        assert "15 more" in body

    def test_a_failed_send_is_logged_not_swallowed(self, engine, caplog):
        """dispatch_informational returns False and never raises."""
        engine.alerts = Gateway(ok=False)
        add(engine.risk_manager, "NVDA", False)
        asyncio.run(engine._dispatch_confirmation_prompt(None))
        assert "FAILED" in caplog.text
