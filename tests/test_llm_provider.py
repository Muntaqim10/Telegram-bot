"""Provider selection for the LLM.

Both providers speak the OpenAI protocol, so switching must be configuration rather than
a code change -- and a missing key must degrade rather than crash the engine.
"""
import pytest

from src.ai.blind_sentiment import PROVIDERS, BlindSentimentAnalyzer, resolve_llm_provider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER", "LLM_MODEL",
                "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


class TestResolution:
    def test_groq_wins_when_both_keys_are_present(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        name, base_url, key, model = resolve_llm_provider()
        assert name == "groq"
        assert base_url == PROVIDERS["groq"]["base_url"]
        assert key == "g"

    def test_falls_back_to_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        name, base_url, key, _ = resolve_llm_provider()
        assert name == "openrouter"
        assert base_url == PROVIDERS["openrouter"]["base_url"]
        assert key == "o"

    def test_no_key_resolves_to_nothing(self):
        assert resolve_llm_provider() == (None, None, None, None)

    def test_llm_provider_forces_the_choice(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        assert resolve_llm_provider()[0] == "openrouter"

    def test_an_unknown_forced_provider_is_ignored(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("LLM_PROVIDER", "nonsense")
        assert resolve_llm_provider()[0] == "groq"

    def test_forcing_a_provider_without_its_key_yields_nothing(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        assert resolve_llm_provider()[0] is None, \
            "must not silently fall through to a provider that was not asked for"


class TestOverrides:
    def test_model_can_be_overridden(self, monkeypatch):
        """Model names change; a rename must not require a code change."""
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("LLM_MODEL", "some-new-model")
        assert resolve_llm_provider()[3] == "some-new-model"

    def test_base_url_can_be_overridden(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("LLM_BASE_URL", "https://proxy.internal/v1")
        assert resolve_llm_provider()[1] == "https://proxy.internal/v1"


class TestAnalyzer:
    def test_it_configures_itself_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        analyzer = BlindSentimentAnalyzer()
        assert analyzer.provider == "groq"
        assert analyzer.model == PROVIDERS["groq"]["default_model"]
        assert analyzer._llm is not None

    def test_explicit_arguments_still_win(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        analyzer = BlindSentimentAnalyzer(api_key="explicit",
                                          base_url="https://example/v1",
                                          model="explicit-model")
        assert analyzer.model == "explicit-model"

    def test_no_key_disables_the_client_without_crashing(self):
        """Catalyst synthesis is decoration; the engine must still alert without it."""
        analyzer = BlindSentimentAnalyzer()
        assert analyzer._llm is None
        assert analyzer.provider == "unconfigured"

    def test_the_positional_call_still_works(self, monkeypatch):
        """main.py used to pass the key positionally; older callers must not break."""
        monkeypatch.setenv("GROQ_API_KEY", "g")
        analyzer = BlindSentimentAnalyzer("a-key")
        assert analyzer._llm is not None
