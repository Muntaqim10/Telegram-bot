### 1. Intraday (Day Trading)
Intraday strategies execute and close positions within the same day.
- **Backtest Period:** 1 to 2 years of recent historical data.
- **Granularity:** 1-minute, 5-minute, or 15-minute intervals.
- **Rule of Thumb:** You need to simulate 300 to 500 intraday trades to get a valid statistical sample. Avoid using too much older historical data, as intraday market microstructure (spreads and volatility) shifts rapidly.

### 2. Swing Trading
Swing trading involves holding positions for several days to a few weeks to capture short- to medium-term price moves.
- **Backtest Period:** 5 to 10 years.
- **Granularity:** Daily (D1) charts are standard, utilizing 1-hour or 4-hour charts for precise entry and exit.
- **Rule of Thumb:** Your backtest must include diverse market environments—bull runs, bear markets, and sideways chop. You should aim for a minimum of 100 to 150 trades.

### 3. LEAPS (Long-Term Equity Anticipation Securities)
LEAP strategies use options with expirations ranging from several months to years.
- **Backtest Period:** 10+ years (at least one full economic or business cycle).
- **Granularity:** Weekly (W1) or Daily (D1) charts.
- **Rule of Thumb:** Because LEAP holding times can span years, testing depth is critical. You must test your strategy through major macro events (like recessions, inflation spikes, and rapid corrections) to understand how multi-year time decay and volatility shifts impact the option's Greeks.
