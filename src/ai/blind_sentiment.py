import json
import logging
from typing import List, Tuple
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

class BlindSentimentAnalyzer:
    """
    Objectively scores financial news sentiment without any knowledge of the current
    technical setup or intended trade direction (Long/Short). This prevents confirmation bias.
    """
    def __init__(self, api_key: str):
        self._llm = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key) if api_key else None

    async def score_headlines(self, ticker: str, headlines: List[str]) -> Tuple[float, str, bool, float]:
        """
        Analyzes a list of headlines and returns:
        (score: float, catalyst: str, is_whale: bool, confidence: float)
        Score ranges from 0.0 (extremely bearish) to 1.0 (extremely bullish), where 0.5 is neutral.
        """
        if not self._llm:
            return 0.5, "AI Offline", False, 0.0
            
        if not headlines:
            return 0.5, "No Recent Catalysts", False, 0.0
            
        prompt = (
            f"Perform a professional, objective analysis of the following institutional catalysts for ${ticker}.\n\n"
            f"CRITICAL RULES:\n"
            f"1. Return ONLY a SINGLE valid JSON object representing the overall aggregated sentiment. DO NOT return a list or array.\n"
            f"2. Analyze the context strictly from the perspective of an objective equity analyst.\n"
            f"3. Output schema must contain 'score' (-1.00 to 1.00), 'conf' (0.00 to 1.00), 'catalyst' (string), and 'whale_detected' (boolean).\n"
            f"4. score: Directional Sentiment Intensity from -1.00 (extremely bearish/negative news) to 1.00 (extremely bullish/positive news). 0.00 means neutral noise.\n"
            f"5. conf: High Confidence (0.80 - 1.00) means explicit earnings beats/misses or major contract wins/losses. Medium (0.40 - 0.79) means analyst upgrades/downgrades or rumors. Low (0.00 - 0.39) means generic noise/opinion.\n"
            f"6. Institutional Whales: Set 'whale_detected' to true only if headlines report SEC filings, major dark pool purchases, heavy block trades, or massive insider transactions.\n\n"
            f"Headlines:\n" + "\n".join(headlines[:10])
        )
        
        try:
            resp = await self._llm.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[
                    {"role": "system", "content": f"You are an objective, deterministic financial text processing engine. Your sole task is to extract and quantify fundamental market sentiment for the target ticker {ticker} from raw financial news headlines. Strictly verify that the headlines correspond to {ticker}. Do not attribute news of other companies to {ticker}. Return ONLY a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0, # Force maximum determinism
                timeout=10.0
            )
            
            content = resp.choices[0].message.content
            data = json.loads(content)
            
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    data = data[0]
                else:
                    data = {}
            elif not isinstance(data, dict):
                data = {}
            
            raw_score = float(data.get('score', 0.0))
            # Normalize from [-1.0, 1.0] scale to [0.0, 1.0] scale (0.50 is neutral)
            score = (raw_score + 1.0) / 2.0
            
            conf = float(data.get('conf', 0.0))
            is_whale = bool(data.get('whale_detected', False))
            
            # Apply confidence dampener: if confidence is extremely low, pull score towards 0.5 (Neutral)
            if conf < 0.40:
                score = 0.5 + ((score - 0.5) * conf)
                
            return score, data.get('catalyst', 'Unknown'), is_whale, conf
            
        except Exception as e:
            log.warning(f"OpenRouter sentiment scoring failed for {ticker}: {e}")
            return 0.5, "AI Error", False, 0.0
