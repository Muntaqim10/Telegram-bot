"""Hard loss limits, enforced by the bot rather than by willpower.

This module encodes one constraint: a ceiling on what may be lost in a period, and a
ceiling on what may be risked in a single trade. When either is breached the bot stops
sending entry alerts. Exit alerts keep firing, because a position already open still
needs watching.

Nothing here improves returns. It bounds losses, which is the only half of "I need gains
and cannot afford losses" that software can actually deliver.

Every limit is opt-in and set by the trader. Defaults are drawn from this account's own
history rather than invented: median premium paid per contract across 322 contracts
(2019-2026) was $174, mean $471, and every one of the ten worst single-trade losses came
from an entry several times larger than that median.

Configure via environment variables:

    RISK_MAX_LOSS_PER_WEEK      dollars; blank or 0 disables the weekly circuit breaker
    RISK_MAX_LOSS_PER_MONTH     dollars; blank or 0 disables the monthly circuit breaker
    RISK_MAX_PREMIUM_PER_TRADE  dollars of premium in one contract; 0 disables
    RISK_MAX_OPEN_POSITIONS     concurrent confirmed positions; 0 disables
"""
import datetime
import logging
import os
import sqlite3

log = logging.getLogger(__name__)

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/rallyhunter.db"))


def _env_float(name: str, default: float = 0.0) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw.replace("$", "").replace(",", ""))
    except ValueError:
        log.warning(f"{name}={raw!r} is not a number; treating the limit as disabled.")
        return default


class RiskBudget:
    """Tracks realized losses against trader-set ceilings and halts entry alerts."""

    def __init__(self, db_path: str = DB_PATH,
                 max_loss_week: float = None, max_loss_month: float = None,
                 max_premium_per_trade: float = None, max_open_positions: int = None):
        self.db_path = db_path
        self.max_loss_week = (max_loss_week if max_loss_week is not None
                              else _env_float("RISK_MAX_LOSS_PER_WEEK"))
        self.max_loss_month = (max_loss_month if max_loss_month is not None
                               else _env_float("RISK_MAX_LOSS_PER_MONTH"))
        self.max_premium_per_trade = (max_premium_per_trade if max_premium_per_trade is not None
                                      else _env_float("RISK_MAX_PREMIUM_PER_TRADE"))
        self.max_open_positions = int(max_open_positions if max_open_positions is not None
                                      else _env_float("RISK_MAX_OPEN_POSITIONS"))
        self._halt_announced_on = None

    # ---------------------------------------------------------------- realized P&L

    def realized_pnl_since(self, since: datetime.date) -> float:
        """Net P&L on contracts CLOSED on or after `since`, from the statement import
        and from positions closed via /closed. Open positions are excluded: an unrealized
        loss is not yet a loss, and counting it would halt on noise."""
        if not os.path.exists(self.db_path):
            return 0.0
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM real_fills "
                "WHERE close_date IS NOT NULL AND close_date >= ? AND pnl IS NOT NULL",
                (since.isoformat(),),
            ).fetchone()
            conn.close()
            return float(row[0] or 0.0)
        except sqlite3.Error as e:
            log.warning(f"Could not read realized P&L: {e}")
            return 0.0

    def status(self, today: datetime.date = None) -> dict:
        """Current standing against every configured limit."""
        today = today or datetime.date.today()
        week_start = today - datetime.timedelta(days=today.weekday())
        month_start = today.replace(day=1)

        week_pnl = self.realized_pnl_since(week_start)
        month_pnl = self.realized_pnl_since(month_start)

        breaches = []
        if self.max_loss_week > 0 and week_pnl <= -self.max_loss_week:
            breaches.append(f"weekly loss ${-week_pnl:,.2f} has reached the "
                            f"${self.max_loss_week:,.2f} limit")
        if self.max_loss_month > 0 and month_pnl <= -self.max_loss_month:
            breaches.append(f"monthly loss ${-month_pnl:,.2f} has reached the "
                            f"${self.max_loss_month:,.2f} limit")

        return {
            "week_start": week_start.isoformat(),
            "month_start": month_start.isoformat(),
            "week_pnl": round(week_pnl, 2),
            "month_pnl": round(month_pnl, 2),
            "max_loss_week": self.max_loss_week,
            "max_loss_month": self.max_loss_month,
            "max_premium_per_trade": self.max_premium_per_trade,
            "max_open_positions": self.max_open_positions,
            "breaches": breaches,
            "halted": bool(breaches),
            "any_limit_set": any([self.max_loss_week, self.max_loss_month,
                                  self.max_premium_per_trade, self.max_open_positions]),
        }

    # ---------------------------------------------------------------- entry gate

    def check_entry(self, premium: float = None, open_positions: int = 0,
                    today: datetime.date = None) -> dict:
        """Whether a new entry alert may be sent.

        Returns {"allowed": bool, "reason": str|None, "status": dict}. A blocked entry is
        a suppressed alert, not a suppressed exit: positions already open keep their
        expiration and stop warnings regardless.
        """
        st = self.status(today)
        if st["breaches"]:
            return {"allowed": False, "reason": "; ".join(st["breaches"]), "status": st}

        if self.max_open_positions > 0 and open_positions >= self.max_open_positions:
            return {"allowed": False,
                    "reason": f"{open_positions} positions already open, limit is "
                              f"{self.max_open_positions}",
                    "status": st}

        if self.max_premium_per_trade > 0 and premium and premium > self.max_premium_per_trade:
            return {"allowed": False,
                    "reason": f"premium ${premium:,.2f} exceeds the per-trade cap of "
                              f"${self.max_premium_per_trade:,.2f}",
                    "status": st}

        return {"allowed": True, "reason": None, "status": st}

    def should_announce_halt(self, today: datetime.date = None) -> bool:
        """True once per day while halted, so the trader is told rather than left
        wondering why the alerts went quiet."""
        today = today or datetime.date.today()
        if not self.status(today)["halted"]:
            self._halt_announced_on = None
            return False
        if self._halt_announced_on == today:
            return False
        self._halt_announced_on = today
        return True

    # ---------------------------------------------------------------- reporting

    def describe(self, today: datetime.date = None) -> str:
        """Telegram-ready summary for /budget."""
        st = self.status(today)
        if not st["any_limit_set"]:
            return (
                "<b>💰 Risk Budget</b>\n\n"
                "<i>No limits set — the bot will not stop you.</i>\n\n"
                "Set these in your environment to turn the circuit breaker on:\n"
                "• <code>RISK_MAX_LOSS_PER_WEEK</code>\n"
                "• <code>RISK_MAX_LOSS_PER_MONTH</code>\n"
                "• <code>RISK_MAX_PREMIUM_PER_TRADE</code>\n"
                "• <code>RISK_MAX_OPEN_POSITIONS</code>\n\n"
                "Pick the numbers yourself — they depend on money only you can see."
            )

        def line(label, pnl, limit):
            if limit <= 0:
                return f"• {label}: <code>${pnl:+,.2f}</code> <i>(no limit set)</i>"
            used = min(100.0, max(0.0, -pnl / limit * 100)) if limit else 0.0
            bar = "█" * int(used / 10) + "░" * (10 - int(used / 10))
            return (f"• {label}: <code>${pnl:+,.2f}</code> of <code>-${limit:,.2f}</code> "
                    f"{bar} {used:.0f}%")

        lines = ["<b>💰 Risk Budget</b>", ""]
        lines.append(line(f"This week (from {st['week_start']})", st["week_pnl"], st["max_loss_week"]))
        lines.append(line(f"This month (from {st['month_start']})", st["month_pnl"], st["max_loss_month"]))
        if st["max_premium_per_trade"] > 0:
            lines.append(f"• Max premium per trade: <code>${st['max_premium_per_trade']:,.2f}</code>")
        if st["max_open_positions"] > 0:
            lines.append(f"• Max open positions: <code>{st['max_open_positions']}</code>")
        lines.append("")
        if st["halted"]:
            lines.append("🛑 <b>ENTRY ALERTS HALTED</b>")
            for b in st["breaches"]:
                lines.append(f"   {b}")
            lines.append("")
            lines.append("<i>Exit warnings on open positions keep running.</i>")
        else:
            lines.append("✅ Within budget. Entry alerts active.")
        lines.append("")
        lines.append("<i>Realized P&amp;L only, from closed contracts.</i>")
        return "\n".join(lines)
