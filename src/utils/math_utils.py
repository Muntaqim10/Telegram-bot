import os
import math

def calculate_take_profit(entry_price: float, atr: float, direction: str) -> float:
    """
    Calculates a take-profit target scaled to the expected hold duration.
    Volatility scales with the square root of time. A realistic target for an N-day hold
    is the daily ATR multiplied by sqrt(N). This prevents unrealistic profit targets
    that options contracts might decay before reaching.
    """
    expected_hold_days = int(os.getenv("HOLD_DAYS", "3"))
    hold_scaled_target = atr * math.sqrt(expected_hold_days)
    
    if direction == "Long":
        return entry_price + hold_scaled_target
    else:
        return entry_price - hold_scaled_target
