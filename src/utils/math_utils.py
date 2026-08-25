import os
import math

def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def black_scholes_price(S: float, K: float, T: float, r: float = 0.045, sigma: float = 0.35, option_type: str = "call") -> float:
    """Calculates Black-Scholes theoretical price."""
    if T <= 0.0001:
        return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)
    if sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return max(0.01, price)

def black_scholes_delta(S: float, K: float, T: float, r: float = 0.045, sigma: float = 0.35, option_type: str = "call") -> float:
    """Calculates Black-Scholes Delta."""
    if T <= 0.0001 or sigma <= 0:
        if option_type.lower() == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
            
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type.lower() == "call":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0

def black_scholes_theta(S: float, K: float, T: float, r: float = 0.045, sigma: float = 0.35, option_type: str = "call") -> float:
    """Calculates Black-Scholes daily Theta decay."""
    if T <= 0.0001 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    term1 = -(S * norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type.lower() == "call":
        term2 = -r * K * math.exp(-r * T) * norm_cdf(d2)
        return (term1 + term2) / 365.0
    else:
        term2 = r * K * math.exp(-r * T) * norm_cdf(-d2)
        return (term1 + term2) / 365.0

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
