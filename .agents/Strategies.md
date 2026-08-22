# Algorithmic Trading Strategy Specifications

This document outlines the core operational logic, market conditions, indicator sets, mathematical formulas, and inherent risks for the primary algorithmic trading frameworks active in this system.

---

## 1. Breakout & Breakdown Strategies
* **Classification:** Momentum / Trend-following (Long & Short)
* **Status:** **ACTIVE**
* **Core Philosophy:** Asset prices breaking past established support or resistance levels signal the initiation or continuation of a strong directional trend, supported by volume confirmation.
* **Mathematical Specification:**
  * Let $H_i$ and $L_i$ be the High and Low prices for day $i$.
  * **20-Day Resistance ($R_{20d}$):** $\max_{t-20 \le i \le t-1}(H_i)$ (incorporating yesterday)
  * **20-Day Support ($S_{20d}$):** $\min_{t-20 \le i \le t-1}(L_i)$
  * Let $CHOP_{14}$ be the 14-period Choppiness Index.
  * Let $RSI_{14}$ be the 14-period RSI.
  * **Volume Z-Score ($Z_{vol}$):** $\frac{V_{projected} - \mu_{v}}{\sigma_{v} + 1e-9}$ where $V_{projected}$ is today's projected intraday volume, and $\mu_{v}$ and $\sigma_{v}$ are the volume mean and standard deviation over 20 completed sessions.
  
* **Triggers:**
  * **Trend Breakout (Long):** $Close_{t} > R_{20d}$ AND $Z_{vol} \ge 1.5$ AND $Close_{t} > SMA_{20} > SMA_{60}$ AND $CHOP_{14} \le 55.0$ AND $RSI_{14} \le 90$
  * **Trend Breakdown (Short):** $Close_{t} < S_{20d}$ AND $Z_{vol} \ge 1.5$ AND $Close_{t} < SMA_{20} < SMA_{60}$ AND $CHOP_{14} \le 55.0$ AND $RSI_{14} \ge 10$
  * **Volume Breakout (Long):** $Close_{t} > R_{20d}$ AND $Z_{vol} \ge 2.0$ AND $Close_{t} > SMA_{20} > SMA_{60}$ AND $CHOP_{14} \le 55.0$ AND $RSI_{14} \le 90$
  * **Volume Breakdown (Short):** $Close_{t} < S_{20d}$ AND $Z_{vol} \ge 2.0$ AND $Close_{t} < SMA_{20} < SMA_{60}$ AND $CHOP_{14} \le 55.0$ AND $RSI_{14} \ge 10$
  * **Intraday ORB Breakout (Long):** Price breaks above the 15-minute Opening Range High, with $Z_{vol} \ge 1.5$ (Trend) or $\ge 2.0$ (Volume), $Price > VWAP$, and $50 < RSI_{14} \le 90$, restricted to the 10:00 AM – 11:30 AM EST window.
  * **Intraday ORB Breakdown (Short):** Price breaks below the 15-minute Opening Range Low, with $Z_{vol} \ge 1.5$ (Trend) or $\ge 2.0$ (Volume), $Price < VWAP$, and $10 \le RSI_{14} < 50$, restricted to the 10:00 AM – 11:30 AM EST window.

* **Exit Parameters:**
  * **Long Exit:** Stop Loss = $Price - 1.5 \times ATR$, Take Profit = $Price + 2.0 \times ATR$
  * **Short Exit:** Stop Loss = $Price + 1.5 \times ATR$, Take Profit = $Price - 2.0 \times ATR$
  * *Trailing:* Trails by $1.5 \times ATR$ once price moves $+1.0 \times ATR$ in the trade direction.

* **Optimal Market Conditions:** High-momentum, strongly trending environments with clear directional biases.
* **Primary System Risks:** Whipsaws and fakeouts (mitigated by volume confirmation and $CHOP$ bounds).

## Strategy Comparison Matrix

| Algorithmic Vector | Breakouts (Long & Short) | Breakdowns (Long & Short) |
| :--- | :--- | :--- |
| **System Status** | **ACTIVE** | **ACTIVE** |
| **Market Outlook** | Price will expand upward out of its current range. | Price will collapse downward below support. |
| **Primary Indicators** | Donchian Channels ($R_{20d}$), Volume ($Z_{vol} \ge 1.5$), $SMA_{20}$, $SMA_{60}$, $CHOP_{14}$, $RSI_{14}$, $ATR$. | Donchian Channels ($S_{20d}$), Volume ($Z_{vol} \ge 1.5$), $SMA_{20}$, $SMA_{60}$, $CHOP_{14}$, $RSI_{14}$, $ATR$. |
| **Exit Mechanism** | SL: $1.5 \times ATR$<br>TP: $2.0 \times ATR$ | SL: $1.5 \times ATR$<br>TP: $2.0 \times ATR$ |