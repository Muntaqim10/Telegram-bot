"""LLM failover between OpenRouter and Groq.

OpenRouter is out of credits and Groq's free tier is rate-limited, so either alone will
drop synthesis. Running both means one failing does not silence the catalyst text.
"""
import asyncio
import time

import pytest

from src.ai.blind_sentiment import (COOLDOWN_SECONDS, BlindSentimentAnalyzer,
                                    resolve_llm_providers)


class Boom(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class FakeCompletions:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        msg = type("m", (), {"content": self.outcome})()
        return type("r", (), {"choices": [type("c", (), {"message": msg})()]})()


class FakeClient:
    def __init__(self, outcome):
        self.chat = type("chat", (), {"completions": FakeCompletions(outcome)})()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clear every provider variable, or a key present on the developer's machine
    silently changes which chain these tests build."""
    for var in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
                "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_PROVIDER_ORDER",
                "LLM_ANTHROPIC_MODEL"):
        monkeypatch.delenv(var, raising=False)


def analyzer_with(*outcomes):
    """An analyzer whose chain is the given outcomes, in order."""
    a = BlindSentimentAnalyzer.__new__(BlindSentimentAnalyzer)
    a._chain = [{"name": f"p{i}", "kind": "openai", "model": "m",
                 "client": FakeClient(o), "unavailable_until": 0.0}
                for i, o in enumerate(outcomes)]
    a.provider = a._chain[0]["name"] if a._chain else "unconfigured"
    a.model = "m" if a._chain else None
    a._llm = a._chain[0]["client"] if a._chain else None
    a._cache, a.cache_ttl = {}, 300
    return a


class TestChainResolution:
    def test_both_keys_build_a_two_step_chain(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        chain = resolve_llm_providers()
        assert [c["name"] for c in chain] == ["groq", "openrouter"]

    def test_one_key_builds_a_one_step_chain(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        assert [c["name"] for c in resolve_llm_providers()] == ["openrouter"]

    def test_no_key_builds_nothing(self):
        assert resolve_llm_providers() == []

    def test_forcing_a_provider_disables_failover(self, monkeypatch):
        """Pinning one provider is a deliberate choice, not a preference order."""
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        assert [c["name"] for c in resolve_llm_providers()] == ["openrouter"]

    def test_model_override_applies_to_the_first_only(self, monkeypatch):
        """It exists to correct a renamed model, not to rename the whole chain."""
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        monkeypatch.setenv("LLM_MODEL", "pinned")
        chain = resolve_llm_providers()
        assert chain[0]["model"] == "pinned"
        assert chain[1]["model"] != "pinned"


class TestAnthropicInTheChain:
    def test_all_three_keys_build_the_default_order(self, monkeypatch):
        """Groq first because it is free; the paid providers only see overflow."""
        for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.setenv(var, "k")
        assert [c["name"] for c in resolve_llm_providers()] == \
            ["groq", "anthropic", "openrouter"]

    def test_the_order_is_configurable(self, monkeypatch):
        for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.setenv(var, "k")
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,groq")
        assert [c["name"] for c in resolve_llm_providers()] == ["anthropic", "groq"]

    def test_an_unknown_name_in_the_order_is_dropped(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "k")
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "nonsense,groq")
        assert [c["name"] for c in resolve_llm_providers()] == ["groq"]

    def test_anthropic_is_marked_as_its_own_protocol(self, monkeypatch):
        """It does not speak the OpenAI wire format, so it needs its own client."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        entry = resolve_llm_providers()[0]
        assert entry["kind"] == "anthropic"
        assert entry["model"] == "claude-opus-5"

    def test_the_anthropic_model_has_its_own_override(self, monkeypatch):
        """Cost control: this call is a short extraction fired once per alert."""
        monkeypatch.setenv("GROQ_API_KEY", "k")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("LLM_ANTHROPIC_MODEL", "claude-haiku-4-5")
        chain = resolve_llm_providers()
        anthropic_entry = next(c for c in chain if c["name"] == "anthropic")
        assert anthropic_entry["model"] == "claude-haiku-4-5"

    def test_a_refusal_is_treated_as_a_failure_not_a_result(self):
        """stop_reason 'refusal' returns HTTP 200; reading content regardless would
        hand the pipeline an empty catalyst instead of failing over."""
        entry = {"client": None, "model": "claude-opus-5"}
        resp = type("r", (), {
            "stop_reason": "refusal",
            "stop_details": type("d", (), {"category": "cyber"})(),
            "content": [],
        })()

        class FakeMessages:
            async def create(self, **kwargs):
                return resp
        entry["client"] = type("c", (), {"messages": FakeMessages()})()

        with pytest.raises(RuntimeError, match="declined"):
            asyncio.run(BlindSentimentAnalyzer._call_anthropic(entry, "sys", "user"))


class TestFailover:
    def test_the_first_provider_is_used_when_it_works(self):
        a = analyzer_with("ok", "unused")
        resp, served = asyncio.run(a._complete("sys", "user"))
        assert (resp, served) == ("ok", "p0")
        assert a._chain[1]["client"].chat.completions.calls == 0

    def test_it_falls_through_to_the_second(self):
        a = analyzer_with(Boom("insufficient credits", 402), "ok")
        resp, served = asyncio.run(a._complete("sys", "user"))
        assert (resp, served) == ("ok", "p1")

    def test_every_provider_failing_raises(self):
        a = analyzer_with(Boom("down", 500), Boom("also down", 500))
        with pytest.raises(Boom):
            asyncio.run(a._complete("sys", "user"))

    def test_no_provider_configured_raises_clearly(self):
        a = analyzer_with()
        a._chain = []
        with pytest.raises(RuntimeError, match="No LLM provider"):
            asyncio.run(a._complete("sys", "user"))


class TestCooldown:
    def test_a_billing_failure_cools_off_for_a_long_time(self):
        a = analyzer_with(Boom("Insufficient credits", 402), "ok")
        asyncio.run(a._complete("sys", "user"))
        remaining = a._chain[0]["unavailable_until"] - time.time()
        assert remaining > COOLDOWN_SECONDS["transient"], \
            "402 will not clear in a minute; do not retry it on every alert"

    def test_a_transient_failure_cools_off_briefly(self):
        a = analyzer_with(Boom("rate limited", 429), "ok")
        asyncio.run(a._complete("sys", "user"))
        remaining = a._chain[0]["unavailable_until"] - time.time()
        assert 0 < remaining <= COOLDOWN_SECONDS["transient"]

    def test_a_cooling_provider_is_skipped_entirely(self):
        a = analyzer_with(Boom("down", 402), "ok")
        asyncio.run(a._complete("sys", "user"))
        first_calls = a._chain[0]["client"].chat.completions.calls

        asyncio.run(a._complete("sys", "user"))
        assert a._chain[0]["client"].chat.completions.calls == first_calls, \
            "a cooling provider must not be retried"

    def test_recovery_clears_the_cooldown(self):
        a = analyzer_with("ok")
        a._chain[0]["unavailable_until"] = time.time() - 1   # expired
        asyncio.run(a._complete("sys", "user"))
        assert a._chain[0]["unavailable_until"] == 0.0

    @pytest.mark.parametrize("message,status,expected", [
        ("Insufficient credits", 402, "fatal"),
        ("invalid api key", None, "fatal"),
        ("unauthorized", 401, "fatal"),
        ("rate limit exceeded", 429, "transient"),
        ("gateway timeout", 504, "transient"),
        ("connection reset", None, "transient"),
    ])
    def test_failure_classification(self, message, status, expected):
        assert BlindSentimentAnalyzer._failure_kind(Boom(message, status)) == expected
