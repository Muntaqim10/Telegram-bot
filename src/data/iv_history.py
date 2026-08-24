import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Tuple, Dict

log = logging.getLogger(__name__)

class IVHistoryTracker:
    def __init__(self, filepath: str = "data/iv_history.json"):
        self.filepath = filepath
        self._cache: Dict[str, List[Tuple[str, float]]] = {}
        self.load()

    def record_iv(self, ticker: str, iv: float, date: Optional[str] = None) -> None:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        if ticker not in self._cache:
            self._cache[ticker] = []
            
        history = self._cache[ticker]
        
        # Deduplicate: if the last entry is for the same date, update it rather than duplicate
        if history and history[-1][0] == date:
            history[-1] = (date, iv)
        else:
            history.append((date, iv))
            
        # Trim to the most recent 252 entries (~1 trading year)
        if len(history) > 252:
            self._cache[ticker] = history[-252:]
            
        # Auto-save after recording
        self.save()

    def get_iv_rank(self, ticker: str, current_iv: float) -> Optional[float]:
        history = self._cache.get(ticker, [])
        if len(history) < 20:
            return None
            
        historical_ivs = [iv for _, iv in history]
        min_iv = min(historical_ivs)
        max_iv = max(historical_ivs)
        
        if max_iv == min_iv:
            return 50.0 # Prevent division by zero if all historical IVs are identical
            
        iv_rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
        return iv_rank

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save IV history to {self.filepath}: {e}")

    def load(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    # JSON stores tuples as lists, convert them back to tuples
                    self._cache = {
                        ticker: [(item[0], float(item[1])) for item in history]
                        for ticker, history in data.items()
                    }
            except Exception as e:
                log.error(f"Failed to load IV history from {self.filepath}: {e}")
                self._cache = {}
