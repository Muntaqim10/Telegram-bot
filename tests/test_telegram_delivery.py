"""Telegram delivery: pacing, retries, and never failing silently.

A live run dispatched 22 alerts in one second, which is 66 API calls across three chats
against a limit of roughly one per second per chat. Telegram answered 429, the code
recorded `success = False` with no log line, and the alerts vanished. An alerter that
drops alerts without saying so is worse than one that does not alert.
"""
import asyncio

import pytest

from src.alerts import AlertGateway


class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status = status
        self._payload = payload or {}
        self._text = text or f"HTTP {status}"

    async def json(self):
        return self._payload

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Replays a scripted sequence of responses and records send times."""

    def __init__(self, *statuses):
        self.statuses = list(statuses)
        self.calls = []

    def post(self, url, json=None, **kw):
        self.calls.append((asyncio.get_event_loop().time(), json["chat_id"]))
        nxt = self.statuses.pop(0) if self.statuses else 200
        if isinstance(nxt, tuple):
            status, retry_after = nxt
            return FakeResponse(status, {"parameters": {"retry_after": retry_after}})
        return FakeResponse(nxt)


@pytest.fixture
def gateway(monkeypatch):
    gw = AlertGateway(None)
    gw._token = "test-token"
    gw._chat_id = "111"
    gw._url = "https://api.telegram.org/bottest-token/"
    return gw


def run(coro):
    return asyncio.run(coro)


def test_a_successful_send_reports_success(gateway, monkeypatch):
    session = FakeSession(200)
    monkeypatch.setattr(gateway, "get_session", lambda: _wrap(session))
    assert run(gateway._send_telegram("hello")) is True
    assert len(session.calls) == 1


def test_a_rate_limit_is_retried_not_dropped(gateway, monkeypatch, caplog):
    session = FakeSession((429, 0.01), 200)
    monkeypatch.setattr(gateway, "get_session", lambda: _wrap(session))
    monkeypatch.setattr(AlertGateway, "MIN_SECONDS_BETWEEN_SENDS", 0.0)

    assert run(gateway._send_telegram("hello")) is True
    assert len(session.calls) == 2, "the 429 should have been retried"
    assert any("rate-limited" in r.message for r in caplog.records)


def test_a_rejection_is_logged_with_the_reason(gateway, monkeypatch, caplog):
    session = FakeSession(400, 400, 400)
    monkeypatch.setattr(gateway, "get_session", lambda: _wrap(session))
    monkeypatch.setattr(AlertGateway, "MIN_SECONDS_BETWEEN_SENDS", 0.0)

    assert run(gateway._send_telegram("hello")) is False
    assert any("REJECTED" in r.message for r in caplog.records), \
        "a non-200 must never be silent"


def test_persistent_rate_limiting_eventually_gives_up(gateway, monkeypatch):
    session = FakeSession((429, 0.01), (429, 0.01), (429, 0.01))
    monkeypatch.setattr(gateway, "get_session", lambda: _wrap(session))
    monkeypatch.setattr(AlertGateway, "MIN_SECONDS_BETWEEN_SENDS", 0.0)

    assert run(gateway._send_telegram("hello")) is False
    assert len(session.calls) == AlertGateway.MAX_SEND_ATTEMPTS


def test_sends_to_one_chat_are_paced(gateway, monkeypatch):
    session = FakeSession(200, 200)
    monkeypatch.setattr(gateway, "get_session", lambda: _wrap(session))
    monkeypatch.setattr(AlertGateway, "MIN_SECONDS_BETWEEN_SENDS", 0.15)

    async def two_sends():
        await gateway._send_telegram("first")
        await gateway._send_telegram("second")

    run(two_sends())
    gap = session.calls[1][0] - session.calls[0][0]
    assert gap >= 0.14, f"sends were {gap:.3f}s apart; the limit is one per second"


def test_every_configured_chat_receives_it(gateway, monkeypatch):
    gateway._chat_id = "111,222,333"
    session = FakeSession(200, 200, 200)
    monkeypatch.setattr(gateway, "get_session", lambda: _wrap(session))
    monkeypatch.setattr(AlertGateway, "MIN_SECONDS_BETWEEN_SENDS", 0.0)

    assert run(gateway._send_telegram("hello")) is True
    assert [c[1] for c in session.calls] == ["111", "222", "333"]


def test_missing_credentials_return_false(gateway):
    gateway._token = None
    assert run(gateway._send_telegram("hello")) is False


async def _wrap(session):
    return session
