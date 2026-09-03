import json
import logging
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

# Both providers speak the OpenAI protocol, so switching is configuration rather than
# code. Groq is chosen when GROQ_API_KEY is present; otherwise OpenRouter.
PROVIDERS = {
    # Groq first by default: its free tier costs nothing, so it absorbs the volume and
    # the paid providers only see traffic when it fails. Reorder with LLM_PROVIDER_ORDER.
    "groq": {
        "kind": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    # Anthropic speaks its own protocol -- the official SDK, not an OpenAI-compatible
    # shim. Opus 5 is the default; set LLM_ANTHROPIC_MODEL to claude-haiku-4-5 for a
    # much cheaper run, since this call is a short JSON extraction fired once per alert.
    "anthropic": {
        "kind": "anthropic",
        "base_url": None,
        "key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-5",
        "model_env": "LLM_ANTHROPIC_MODEL",
    },
    "openrouter": {
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "deepseek/deepseek-v3.2",
    },
}

DEFAULT_ORDER = ["groq", "anthropic", "openrouter"]


# How long to stop trying a provider after it fails. A billing or auth failure will not
# fix itself in a minute, so there is no point paying the latency on every alert; a rate
# limit or a blip usually will.
COOLDOWN_SECONDS = {"fatal": 1800, "transient": 60}


def resolve_llm_providers():
    """Every configured provider, in the order they should be tried.

    Returns a list of dicts. Groq first by default because its free tier costs nothing;
    LLM_PROVIDER pins one exclusively. LLM_MODEL and LLM_BASE_URL apply to the first
    entry only -- they exist to correct a renamed model, not to describe a whole chain.
    """
    forced = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if forced in PROVIDERS:
        order = [forced]
    else:
        configured = [n.strip().lower() for n in
                      (os.getenv("LLM_PROVIDER_ORDER") or "").split(",") if n.strip()]
        order = [n for n in configured if n in PROVIDERS] or DEFAULT_ORDER

    chain = []
    for name in order:
        cfg = PROVIDERS[name]
        key = os.getenv(cfg["key_env"])
        if not key:
            continue
        first = not chain
        model = os.getenv(cfg.get("model_env", "")) if cfg.get("model_env") else None
        chain.append({
            "name": name,
            "kind": cfg["kind"],
            "base_url": (os.getenv("LLM_BASE_URL") if first else None) or cfg["base_url"],
            "api_key": key,
            "model": (os.getenv("LLM_MODEL") if first else None) or model or cfg["default_model"],
        })
    return chain


def resolve_llm_provider():
    """The first configured provider, as a tuple. Kept for callers that want one."""
    chain = resolve_llm_providers()
    if not chain:
        return None, None, None, None
    c = chain[0]
    return c["name"], c["base_url"], c["api_key"], c["model"]

class BlindSentimentAnalyzer:
    """
    Objectively scores financial news sentiment and synthesizes fundamental catalysts,
    Black-Scholes pricing math, and historical backtest performance using DeepSeek LLM (OpenRouter).
    Includes an in-memory cache to prevent API overuse.
    """
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,
                 provider: str = None):
        """Builds a failover chain from the environment.

        Explicit arguments pin a single provider, which existing callers and tests rely
        on. Otherwise every provider holding a key is tried in turn, so one being out of
        credits or rate-limited does not silence catalyst synthesis.
        """
        if api_key is not None:
            chain = [{"name": provider or "explicit",
                      "base_url": base_url or PROVIDERS["openrouter"]["base_url"],
                      "api_key": api_key,
                      "model": model or PROVIDERS["openrouter"]["default_model"]}]
        else:
            chain = resolve_llm_providers()

        self._chain = []
        for cfg in chain:
            kind = cfg.get("kind", "openai")
            if kind == "anthropic":
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=cfg["api_key"])
            else:
                client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
            self._chain.append({
                "name": cfg["name"],
                "kind": kind,
                "model": cfg["model"],
                "client": client,
                "unavailable_until": 0.0,
            })

        self._cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300 # 5 minutes

        if self._chain:
            self.provider = self._chain[0]["name"]
            self.model = self._chain[0]["model"]
            self._llm = self._chain[0]["client"]   # kept for callers that reach for it
            log.info("LLM chain: " + " -> ".join(
                f"{c['name']}({c['model']})" for c in self._chain))
        else:
            self.provider, self.model, self._llm = "unconfigured", None, None
            log.warning("No LLM key found (set GROQ_API_KEY or OPENROUTER_API_KEY). "
                        "Catalyst synthesis is disabled; alerts still dispatch.")

    @staticmethod
    def _failure_kind(exc) -> str:
        """Billing and auth failures will not clear on their own; the rest might."""
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (401, 402, 403):
            return "fatal"
        text = str(exc).lower()
        if "insufficient" in text or "payment required" in text or "invalid api key" in text:
            return "fatal"
        return "transient"

    @staticmethod
    async def _call_openai(entry, system: str, user: str) -> str:
        resp = await entry["client"].chat.completions.create(
            model=entry["model"],
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=500,
            timeout=12.0,
        )
        return resp.choices[0].message.content

    @staticmethod
    async def _call_anthropic(entry, system: str, user: str) -> str:
        """Anthropic's own protocol: the system prompt is a top-level parameter, not a
        message, and the reply is a list of content blocks.

        max_tokens is generous because thinking is on by default and shares the budget;
        effort is low because this is a short structured extraction, not analysis.
        """
        resp = await entry["client"].messages.create(
            model=entry["model"],
            max_tokens=2000,
            system=system,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": user}],
            timeout=20.0,
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            detail = getattr(getattr(resp, "stop_details", None), "category", None)
            raise RuntimeError(f"anthropic declined this request (category={detail})")
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise RuntimeError("anthropic returned no text block")

    async def _complete(self, system: str, user: str):
        """Runs the request against each configured provider until one answers.

        Returns (raw_text, provider_name). Raises the last error when every provider is
        exhausted, so the caller's existing handling still applies.
        """
        import time as _time
        if not self._chain:
            raise RuntimeError("No LLM provider configured")

        last_error = None
        tried = []
        for entry in self._chain:
            if entry["unavailable_until"] > _time.time():
                continue
            tried.append(entry["name"])
            try:
                caller = (self._call_anthropic if entry["kind"] == "anthropic"
                          else self._call_openai)
                text = await caller(entry, system, user)
                if entry["unavailable_until"]:
                    log.info(f"LLM provider {entry['name']} recovered.")
                    entry["unavailable_until"] = 0.0
                return text, entry["name"]
            except Exception as e:
                last_error = e
                kind = self._failure_kind(e)
                entry["unavailable_until"] = _time.time() + COOLDOWN_SECONDS[kind]
                log.warning(
                    f"LLM provider {entry['name']} failed ({kind}): {str(e)[:160]}. "
                    f"Skipping it for {COOLDOWN_SECONDS[kind]}s."
                )

        if not tried:
            log.debug("Every LLM provider is cooling off; skipping synthesis this cycle.")
        raise last_error or RuntimeError("All LLM providers are cooling off")

    async def score_headlines(self, ticker: str, headlines: List[str]) -> Tuple[float, str, bool, float]:
        """
        Legacy compatibility wrapper for headline scoring.
        Returns: (score: float, catalyst: str, is_whale: bool, confidence: float)
        """
        synth = await self.synthesize_catalyst_and_edge(
            ticker=ticker,
            direction="Long",
            headlines=headlines,
            backtest_stats={},
            math_context={}
        )
        return synth["score"], synth["catalyst"], synth["is_whale"], synth["confidence"]

    async def synthesize_catalyst_and_edge(
        self,
        ticker: str,
        direction: str,
        headlines: List[str],
        backtest_stats: Dict[str, Any],
        math_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        DeepSeek reasoning synthesis that cross-examines:
        1. Live Breaking News & SEC Filings
        2. Empirical Backtest Statistics (Win rate, Avg ROI)
        3. Black-Scholes Mathematical Context (0.75 Delta strike, Expected move %, Breakeven, Theta, IV rank, GEX confidence, and Multi-Timeframe Confluence)
        """
        now = datetime.now()
        cache_key = f"{ticker}_{direction}"

        # In-memory cache check to prevent API overuse
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if (now - cached["timestamp"]).total_seconds() < self.cache_ttl:
                log.debug(f"DeepSeek synthesis cache hit for {ticker}. Returning cached analysis.")
                return cached["data"]

        if not self._llm:
            return {
                "score": 0.50,
                "confidence": 0.0,
                "catalyst": "AI Offline",
                "ai_thesis": "DeepSeek analysis offline.",
                "is_whale": False,
                "verdict": "NEUTRAL"
            }

        # Format context for DeepSeek
        news_text = "\n".join(headlines[:8]) if headlines else "No major recent breaking headlines."
        
        bt_wr = backtest_stats.get("win_rate", 0.0)
        bt_trades = backtest_stats.get("total_trades", 0)
        bt_avg_roi = backtest_stats.get("avg_return_pct", 0.0)
        bt_tier = backtest_stats.get("asset_tier", "MID-CAP")
        bt_age = backtest_stats.get("backtest_age_days")
        age_note = f" (from a backtest run {bt_age:.0f} days ago)" if bt_age is not None else ""

        exp_move = math_context.get("expected_move_pct", 0.0)
        strike = math_context.get("target_strike", 0.0)
        opt_type = math_context.get("option_type", "CALL")
        opt_ask = math_context.get("option_ask", 0.0)
        delta = math_context.get("delta", 0.75)
        theta = math_context.get("theta", 0.0)
        iv_rank = math_context.get("iv_rank")
        gex_conf = math_context.get("gex_confidence", "STANDARD")
        tf_confluence = math_context.get("timeframe_confluence", "N/A")
        
        iv_rank_str = f"{iv_rank:.0f}/100" if iv_rank is not None else "Insufficient history"

        prompt = (
            f"You are a Senior Quantitative Equity & Derivatives Portfolio Manager.\n"
            f"Synthesize the following live news, mathematical pricing, and historical backtest data for ${ticker}:\n\n"
            f"1. TRADE SETUP & MATH:\n"
            f"   • Proposed Action: {direction.upper()} ({opt_type})\n"
            f"   • Asset Tier: {bt_tier}\n"
            f"   • Mathematical Expected Move: +{exp_move:.1f}%\n"
            f"   • 0.75 Delta ITM Contract: ${strike} {opt_type} @ ${opt_ask:.2f} (Delta: {delta:.2f}, Theta: {theta:.4f}/day)\n"
            f"   • IV Rank (vs this ticker's own history): {iv_rank_str}\n"
            f"   • Dealer Positioning Confidence (GEX): {gex_conf}\n"
            f"   • Multi-Timeframe Confluence Already Confirmed: {tf_confluence}\n\n"
            f"2. HISTORICAL BACKTEST PERFORMANCE{age_note}:\n"
            f"   • Sample Size: {bt_trades} trades\n"
            f"   • Historical Win Rate: {bt_wr}%\n"
            f"   • Historical Avg Option ROI: +{bt_avg_roi}%\n\n"
            f"3. LIVE BREAKING NEWS & SEC FILINGS:\n{news_text}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"• Output MUST be a SINGLE valid JSON object.\n"
            f"• 'score': Directional sentiment intensity from -1.00 (extreme negative news) to +1.00 (extreme positive news).\n"
            f"• 'conf': Confidence level (0.00 to 1.00).\n"
            f"• 'catalyst': Concise summary of the fundamental/news catalyst driving the ticker (max 8 words).\n"
            f"• 'ai_thesis': Exactly ONE concise, professional sentence summarizing why the news catalyst, math, and historical backtest edge support (or caution against) this {direction} trade.\n"
            f"• 'whale_detected': boolean (true if SEC filings, major institutional accumulation, or dark pool blocks are reported).\n"
            f"• 'otm_qualified': boolean (set to true ONLY on rare occasions when massive fundamental news catalyst, expected move >= 15%, and historical backtest edge indicate an explosive breakout capable of reaching OTM strikes).\n"
            f"• 'verdict': 'CONCORDANT' if news agrees with technicals, 'DISCORDANT' if news contradicts trade, or 'NEUTRAL' if no news.\n"
        )

        try:
            raw_text, served_by = await self._complete(
                "You are a quantitative derivatives intelligence agent. "
                "Output ONLY a valid JSON object matching the requested schema.",
                prompt,
            )
            if served_by != self.provider:
                log.info(f"[{ticker}] catalyst synthesis served by fallback provider {served_by}.")

            # Models that lack a JSON response mode sometimes wrap the object in prose or
            # a fenced block; take the outermost object rather than failing the alert.
            text = raw_text.strip()
            if not text.startswith("{"):
                start, end = text.find("{"), text.rfind("}")
                if start == -1 or end <= start:
                    raise ValueError(f"no JSON object in the reply from {served_by}")
                text = text[start:end + 1]

            data = json.loads(text)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]

            raw_score = float(data.get("score", 0.0))
            score = (raw_score + 1.0) / 2.0 # Scale to [0.0, 1.0]
            conf = float(data.get("conf", 0.5))
            catalyst = str(data.get("catalyst", "Technical Momentum"))
            ai_thesis = str(data.get("ai_thesis", "Catalyst and backtest metrics align with setup."))
            is_whale = bool(data.get("whale_detected", False))
            otm_qualified = bool(data.get("otm_qualified", False))
            verdict = str(data.get("verdict", "CONCORDANT")).upper()

            result = {
                "score": score,
                "confidence": conf,
                "conf": conf,
                "catalyst": catalyst,
                "ai_thesis": ai_thesis,
                "is_whale": is_whale,
                "otm_qualified": otm_qualified,
                "verdict": verdict
            }

            # Cache the result
            self._cache[cache_key] = {"data": result, "timestamp": now}
            return result

        except Exception as e:
            log.warning(f"DeepSeek multi-domain synthesis error for {ticker}: {e}")
            return {
                "score": 0.50,
                "confidence": 0.0,
                "catalyst": "News Fetch Active",
                "ai_thesis": "Technical momentum setup with dynamic risk management.",
                "is_whale": False,
                "verdict": "NEUTRAL"
            }
