import os
import sys
import csv
import json
import asyncio
import logging
import statistics
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
config_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(dotenv_path=root_env)
load_dotenv(dotenv_path=config_env)

from src.backtest.stats_reader import BacktestStatsReader
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rallyhunter.review_backtest")

def compute_aggregate_statistics(csv_path: str) -> Optional[Dict[str, Any]]:
    """
    Parses a backtest CSV and calculates cross-ticker aggregate metrics.
    """
    if not os.path.exists(csv_path):
        return None

    rows: List[Dict[str, str]] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    total_trades = len(rows)
    wins = sum(1 for r in rows if r.get("outcome", "").upper() == "WIN")
    losses = sum(1 for r in rows if r.get("outcome", "").upper() == "LOSS")
    expired = sum(1 for r in rows if r.get("outcome", "").upper() == "EXPIRED")
    decided = wins + losses
    overall_win_rate = (wins / decided * 100.0) if decided > 0 else 0.0

    # Group by catalyst_type
    catalyst_stats: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cat = r.get("catalyst_type") or "Unknown"
        if cat not in catalyst_stats:
            catalyst_stats[cat] = {"total": 0, "wins": 0, "losses": 0, "expired": 0}
        catalyst_stats[cat]["total"] += 1
        out = r.get("outcome", "").upper()
        if out == "WIN":
            catalyst_stats[cat]["wins"] += 1
        elif out == "LOSS":
            catalyst_stats[cat]["losses"] += 1
        elif out == "EXPIRED":
            catalyst_stats[cat]["expired"] += 1

    for cat, s in catalyst_stats.items():
        c_decided = s["wins"] + s["losses"]
        s["win_rate"] = round((s["wins"] / c_decided * 100.0) if c_decided > 0 else 0.0, 1)

    # Group by asset_tier
    tier_stats: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        tier = r.get("asset_tier") or "Unknown"
        if tier not in tier_stats:
            tier_stats[tier] = {"total": 0, "wins": 0, "losses": 0}
        tier_stats[tier]["total"] += 1
        out = r.get("outcome", "").upper()
        if out == "WIN":
            tier_stats[tier]["wins"] += 1
        elif out == "LOSS":
            tier_stats[tier]["losses"] += 1

    for tier, s in tier_stats.items():
        t_decided = s["wins"] + s["losses"]
        s["win_rate"] = round((s["wins"] / t_decided * 100.0) if t_decided > 0 else 0.0, 1)

    # Returns, Days Held, Ambiguity
    pnl_list: List[float] = []
    days_list: List[float] = []
    ambiguous_count = 0
    ticker_pnls: Dict[str, float] = {}

    for r in rows:
        ticker = r.get("ticker", "").upper()
        try:
            pnl = float(r.get("option_pnl_pct", 0.0))
            pnl_list.append(pnl)
            ticker_pnls[ticker] = ticker_pnls.get(ticker, 0.0) + pnl
        except (ValueError, TypeError):
            pass

        try:
            days = float(r.get("days_held", 1.0))
            days_list.append(days)
        except (ValueError, TypeError):
            pass

        amb_val = str(r.get("same_day_ambiguous", "")).strip().lower()
        if amb_val in ("true", "1", "yes"):
            ambiguous_count += 1

    avg_pnl = statistics.mean(pnl_list) if pnl_list else 0.0
    median_pnl = statistics.median(pnl_list) if pnl_list else 0.0
    stdev_pnl = statistics.stdev(pnl_list) if len(pnl_list) > 1 else 0.0
    avg_days_held = statistics.mean(days_list) if days_list else 0.0
    ambiguous_pct = (ambiguous_count / total_trades * 100.0) if total_trades > 0 else 0.0

    # Best & Worst Tickers
    sorted_tickers = sorted(ticker_pnls.items(), key=lambda x: x[1], reverse=True)
    best_ticker = sorted_tickers[0] if sorted_tickers else ("N/A", 0.0)
    worst_ticker = sorted_tickers[-1] if sorted_tickers else ("N/A", 0.0)

    return {
        "file_name": os.path.basename(csv_path),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "expired": expired,
        "overall_win_rate": round(overall_win_rate, 1),
        "catalyst_breakdown": catalyst_stats,
        "tier_breakdown": tier_stats,
        "avg_option_pnl_pct": round(avg_pnl, 2),
        "median_option_pnl_pct": round(median_pnl, 2),
        "stdev_option_pnl_pct": round(stdev_pnl, 2),
        "same_day_ambiguous_count": ambiguous_count,
        "same_day_ambiguous_pct": round(ambiguous_pct, 1),
        "avg_days_held": round(avg_days_held, 1),
        "best_ticker": {"ticker": best_ticker[0], "total_pnl_pct": round(best_ticker[1], 1)},
        "worst_ticker": {"ticker": worst_ticker[0], "total_pnl_pct": round(worst_ticker[1], 1)}
    }

def print_console_summary(stats: Dict[str, Any]):
    """Prints a clean human-readable summary of the aggregate backtest statistics."""
    print("=" * 68)
    print(f"  AGGREGATE BACKTEST AUDIT: {stats['file_name']}")
    print("=" * 68)
    print(f"• Total Trades:       {stats['total_trades']} ({stats['wins']} Wins | {stats['losses']} Losses | {stats['expired']} Expired)")
    print(f"• Overall Win Rate:   {stats['overall_win_rate']}% (on decided trades)")
    print(f"• Avg Option Return:  {stats['avg_option_pnl_pct']:+.2f}% (Median: {stats['median_option_pnl_pct']:+.2f}%, StdDev: {stats['stdev_option_pnl_pct']:.2f}%)")
    print(f"• Avg Holding Period: {stats['avg_days_held']:.1f} days")
    print(f"• Ambiguous Exits:    {stats['same_day_ambiguous_count']}/{stats['total_trades']} ({stats['same_day_ambiguous_pct']}%) [Same-Day SL/TP tiebreak]")
    print(f"• Best Ticker:        ${stats['best_ticker']['ticker']} ({stats['best_ticker']['total_pnl_pct']:+.1f}% cumulative PnL)")
    print(f"• Worst Ticker:       ${stats['worst_ticker']['ticker']} ({stats['worst_ticker']['total_pnl_pct']:+.1f}% cumulative PnL)")
    
    print("\n--- Win Rate by Catalyst Type ---")
    for cat, s in stats["catalyst_breakdown"].items():
        print(f"  • {cat:<24}: {s['win_rate']:>5.1f}% Win Rate ({s['wins']}W / {s['losses']}L / {s['total']} Total)")

    print("\n--- Win Rate by Asset Tier ---")
    for tier, s in stats["tier_breakdown"].items():
        print(f"  • {tier:<24}: {s['win_rate']:>5.1f}% Win Rate ({s['wins']}W / {s['losses']}L / {s['total']} Total)")
    print("=" * 68)

async def review_with_deepseek(stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Sends aggregate statistics to DeepSeek via OpenRouter for methodological review."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n⚠️ Note: No OPENROUTER_API_KEY or DEEPSEEK_API_KEY configured. Skipping LLM critique.")
        return None

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    prompt = f"""
Please review and critique the following aggregate backtest methodology and performance statistics for a quantitative ~0.75 Delta ITM options momentum strategy:

1. AGGREGATE PERFORMANCE SUMMARY:
   - Total Trades Tested: {stats['total_trades']}
   - Overall Win Rate: {stats['overall_win_rate']}% ({stats['wins']} Wins, {stats['losses']} Losses, {stats['expired']} Expired)
   - Mean Option Return: {stats['avg_option_pnl_pct']:+.2f}%
   - Median Option Return: {stats['median_option_pnl_pct']:+.2f}%
   - Return Std Deviation: {stats['stdev_option_pnl_pct']:.2f}%
   - Avg Holding Period: {stats['avg_days_held']} days
   - Same-Day Ambiguous Exits: {stats['same_day_ambiguous_count']} ({stats['same_day_ambiguous_pct']}%) [Occurred when bar High >= TP and Low <= SL on same bar]
   - Best Performer: ${stats['best_ticker']['ticker']} ({stats['best_ticker']['total_pnl_pct']:+.1f}% cumulative PnL)
   - Worst Performer: ${stats['worst_ticker']['ticker']} ({stats['worst_ticker']['total_pnl_pct']:+.1f}% cumulative PnL)

2. WIN RATE BY CATALYST TYPE:
{json.dumps(stats['catalyst_breakdown'], indent=2)}

3. WIN RATE BY ASSET TIER:
{json.dumps(stats['tier_breakdown'], indent=2)}

AUDIT TASKS:
1. Identify any statistics that look internally inconsistent, suspicious, or prone to overfitting (e.g. small sample sizes with skewed win rates, large gap between mean and median return, high same-day ambiguous tiebreak rate, or ticker concentration).
2. Suggest 2-3 specific additional quantitative metrics that would help evaluate this strategy's real-world viability beyond what is currently tracked.
3. Provide a concise plain-language summary of your methodological critique.

CRITICAL INSTRUCTIONS:
- You are reviewing the statistical rigor and methodology of a backtest, NOT providing investment advice or trade recommendations.
- Output MUST be a SINGLE valid JSON object with EXACTLY these keys:
  {{
    "consistency_flags": ["<point 1>", "<point 2>"],
    "suggested_additional_metrics": ["<metric 1>", "<metric 2>"],
    "summary": "<one paragraph plain-language summary>"
  }}
"""

    try:
        resp = await client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Quantitative Research Auditor and Methodology Reviewer. "
                        "Your role is strictly to conduct a rigorous statistical and methodological audit of the provided aggregate backtest output. "
                        "You do NOT predict future performance, provide trading advice, or recommend positions. "
                        "Output ONLY a valid JSON object matching the requested schema."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=800,
            timeout=25.0
        )

        content = resp.choices[0].message.content
        data = json.loads(content)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        return data

    except Exception as e:
        print(f"\n❌ DeepSeek review API call failed: {e}")
        return None

async def main():
    reader = BacktestStatsReader()
    if not reader.reload_latest_results() or not reader._last_loaded_file:
        print("❌ No backtest CSV results found in backtest_results/ directory.")
        return

    latest_csv = reader._last_loaded_file
    stats = compute_aggregate_statistics(latest_csv)
    if not stats:
        print(f"❌ Failed to parse backtest statistics from {latest_csv}")
        return

    print_console_summary(stats)

    print("\n🔍 Querying DeepSeek Quantitative Auditor for Methodological Critique...")
    critique = await review_with_deepseek(stats)

    if critique:
        print("\n" + "=" * 68)
        print("  DEEPSEEK METHODOLOGY & STATISTICAL AUDIT CRITIQUE")
        print("=" * 68)
        
        flags = critique.get("consistency_flags") or critique.get("red_flags") or critique.get("issues") or []
        print("\n🚩 Consistency & Methodology Red Flags:")
        if flags:
            for flag in flags:
                print(f"  • {flag}")
        else:
            print("  • No major statistical inconsistencies detected.")

        metrics = critique.get("suggested_additional_metrics") or critique.get("suggested_metrics") or critique.get("additional_metrics") or []
        print("\n💡 Suggested Additional Quantitative Metrics:")
        if metrics:
            for m in metrics:
                print(f"  • {m}")

        summary = critique.get("summary") or critique.get("executive_summary") or critique.get("critique") or ""
        if summary:
            print(f"\n📝 Executive Summary:\n{summary}")
        print("=" * 68)

if __name__ == "__main__":
    asyncio.run(main())
