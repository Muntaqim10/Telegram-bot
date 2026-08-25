import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

class BlindSentimentAnalyzer:
    """
    Objectively scores financial news sentiment and synthesizes fundamental catalysts,
    Black-Scholes pricing math, and historical backtest performance using DeepSeek LLM (OpenRouter).
    Includes an in-memory cache to prevent API overuse.
    """
    def __init__(self, api_key: str):
        self._llm = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key) if api_key else None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300 # 5 minutes

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
        3. Black-Scholes Mathematical Context (0.75 Delta strike, Expected move %, Breakeven)
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

        exp_move = math_context.get("expected_move_pct", 0.0)
        strike = math_context.get("target_strike", 0.0)
        opt_type = math_context.get("option_type", "CALL")
        opt_ask = math_context.get("option_ask", 0.0)
        delta = math_context.get("delta", 0.75)

        prompt = (
            f"You are a Senior Quantitative Equity & Derivatives Portfolio Manager.\n"
            f"Synthesize the following live news, mathematical pricing, and historical backtest data for ${ticker}:\n\n"
            f"1. TRADE SETUP & MATH:\n"
            f"   • Proposed Action: {direction.upper()} ({opt_type})\n"
            f"   • Asset Tier: {bt_tier}\n"
            f"   • Mathematical Expected Move: +{exp_move:.1f}%\n"
            f"   • 0.75 Delta ITM Contract: ${strike} {opt_type} @ ${opt_ask:.2f} (Delta: {delta:.2f})\n\n"
            f"2. HISTORICAL BACKTEST PERFORMANCE:\n"
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
            f"• 'verdict': 'CONCORDANT' if news agrees with technicals, 'DISCORDANT' if news contradicts trade, or 'NEUTRAL' if no news.\n"
        )

        try:
            resp = await self._llm.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative derivatives intelligence agent. Output ONLY a valid JSON object matching the requested schema."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=12.0
            )

            data = json.loads(resp.choices[0].message.content)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]

            raw_score = float(data.get("score", 0.0))
            score = (raw_score + 1.0) / 2.0 # Scale to [0.0, 1.0]
            conf = float(data.get("conf", 0.5))
            catalyst = str(data.get("catalyst", "Technical Momentum"))
            ai_thesis = str(data.get("ai_thesis", "Catalyst and backtest metrics align with setup."))
            is_whale = bool(data.get("whale_detected", False))
            verdict = str(data.get("verdict", "CONCORDANT")).upper()

            result = {
                "score": score,
                "confidence": conf,
                "catalyst": catalyst,
                "ai_thesis": ai_thesis,
                "is_whale": is_whale,
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
                "verdict": "CONCORDANT"
            }
