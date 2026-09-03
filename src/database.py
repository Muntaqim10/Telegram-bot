import asyncio
import csv
import logging
import os
import re
import sqlite3
from typing import Any

log = logging.getLogger(__name__)

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/rallyhunter.db"))
CSV_LIVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trade_log.csv"))
CSV_ARCHIVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trade_log_archive.csv"))

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def _migrate_csv_data(conn):
    """Migrates historical records from CSVs into SQLite tables if they are empty."""
    cursor = conn.cursor()
    
    # 1. Migrate Live Trades
    cursor.execute("SELECT COUNT(*) FROM trade_log")
    if cursor.fetchone()[0] == 0 and os.path.exists(CSV_LIVE_PATH):
        try:
            with open(CSV_LIVE_PATH, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    if len(row) < 11:
                        continue
                    # Handle optional columns safely
                    sl = float(row[7].replace('$', '').strip()) if row[7] else 0.0
                    tp = float(row[8].replace('$', '').strip()) if row[8] else 0.0
                    is_whale = 1 if row[4].strip().lower() == "true" else 0
                    
                    cursor.execute("""
                        INSERT INTO trade_log (
                            timestamp, ticker, price, z_vol, is_whale, growth, win_prob, sl, tp, strategy, direction, catalyst
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row[0], row[1].upper().strip(), float(row[2].replace('$', '').strip()),
                        row[3], is_whale, row[5], row[6], sl, tp, row[9], row[10], row[11] if len(row) > 11 else ""
                    ))
            print("[DB] Migrated live trade log from CSV to SQLite successfully.")
        except Exception as e:
            print(f"[DB _migrate_csv_data] Error migrating live CSV to trade_log table: {e}")

    # 2. Migrate Archived Trades
    cursor.execute("SELECT COUNT(*) FROM trade_log_archive")
    if cursor.fetchone()[0] == 0 and os.path.exists(CSV_ARCHIVE_PATH):
        try:
            with open(CSV_ARCHIVE_PATH, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) < 11:
                        continue
                    sl = float(row[7].replace('$', '').strip()) if row[7] else 0.0
                    tp = float(row[8].replace('$', '').strip()) if row[8] else 0.0
                    is_whale = 1 if row[4].strip().lower() == "true" else 0
                    
                    cursor.execute("""
                        INSERT INTO trade_log_archive (
                            timestamp, ticker, price, z_vol, is_whale, growth, win_prob, sl, tp, strategy, direction, catalyst
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row[0], row[1].upper().strip(), float(row[2].replace('$', '').strip()),
                        row[3], is_whale, row[5], row[6], sl, tp, row[9], row[10], row[11] if len(row) > 11 else ""
                    ))
            print("[DB] Migrated archived trade log from CSV to SQLite successfully.")
        except Exception as e:
            print(f"[DB _migrate_csv_data] Error migrating archived CSV to trade_log_archive table: {e}")
            
    conn.commit()

def _init_db_sync():
    """Synchronously initialize tables in the database and run migrations."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create live trade log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            is_whale INTEGER NOT NULL,
            win_prob TEXT NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL,
            catalyst TEXT NOT NULL,
            xgb_win_prob REAL DEFAULT 0.5,
            sentinel_verdict TEXT DEFAULT 'NOT_CHECKED',
            conviction TEXT DEFAULT '',
            warning_tag TEXT DEFAULT ''
        )
    """)
    
    # Create archived trade log table. exit_price/outcome are first-class columns so a
    # closed trade can be joined back to the prediction that opened it -- they used to be
    # concatenated into the catalyst text, which left the whole log unqueryable.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_log_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            is_whale INTEGER NOT NULL,
            win_prob TEXT NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL,
            catalyst TEXT NOT NULL,
            xgb_win_prob REAL DEFAULT 0.5,
            sentinel_verdict TEXT DEFAULT 'NOT_CHECKED',
            conviction TEXT DEFAULT '',
            warning_tag TEXT DEFAULT '',
            exit_price REAL,
            outcome TEXT
        )
    """)
    conn.commit()
    
    # Migrate existing tables to add new columns if they don't exist
    for table in ['trade_log', 'trade_log_archive']:
        for col, col_type in [("xgb_win_prob", "REAL DEFAULT 0.5"), 
                              ("sentinel_verdict", "TEXT DEFAULT 'NOT_CHECKED'"),
                              ("conviction", "TEXT DEFAULT ''"),
                              ("warning_tag", "TEXT DEFAULT ''")]:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # Column already exists

    # Only closed trades have an exit, so these live on the archive alone.
    for col, col_type in [("exit_price", "REAL"), ("outcome", "TEXT")]:
        try:
            cursor.execute(f"ALTER TABLE trade_log_archive ADD COLUMN {col} {col_type}")
        except Exception:
            pass  # Column already exists
    conn.commit()

    _backfill_archive_outcomes(conn)
    
    # Run CSV migrations
    _migrate_csv_data(conn)
    conn.close()

async def init_db():
    """Asynchronously initialize the database."""
    await asyncio.to_thread(_init_db_sync)

def _add_trade_sync(table: str, trade: dict[str, Any]):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(f"PRAGMA table_info({table})")
    existing_cols = [row["name"] for row in cursor.fetchall()]
    
    insert_cols = []
    insert_vals = []
    
    # Add all provided trade data
    for k, v in trade.items():
        insert_cols.append(k)
        insert_vals.append(v)
        
    # Handle legacy columns for old schema
    if "z_vol" in existing_cols and "z_vol" not in trade:
        insert_cols.append("z_vol")
        insert_vals.append("")
    if "growth" in existing_cols and "growth" not in trade:
        insert_cols.append("growth")
        insert_vals.append("")
        
    col_str = ", ".join(insert_cols)
    val_str = ", ".join(["?"] * len(insert_vals))
    
    cursor.execute(f"INSERT INTO {table} ({col_str}) VALUES ({val_str})", tuple(insert_vals))
    conn.commit()
    conn.close()

async def add_trade(trade: dict[str, Any]):
    """Add a trade to the live trade log."""
    await asyncio.to_thread(_add_trade_sync, "trade_log", trade)

async def archive_trade(trade: dict[str, Any]):
    """Add a trade to the archived trade log."""
    await asyncio.to_thread(_add_trade_sync, "trade_log_archive", trade)

def _get_trades_sync(table: str) -> list[dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table} ORDER BY timestamp ASC")
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        row_dict = dict(r)
        trade_obj = {
            "id": row_dict["id"],
            "timestamp": row_dict["timestamp"],
            "ticker": row_dict["ticker"],
            "price": row_dict["price"],
            "is_whale": bool(row_dict["is_whale"]),
            "win_prob": row_dict["win_prob"],
            "sl": row_dict["sl"],
            "tp": row_dict["tp"],
            "strategy": row_dict["strategy"],
            "direction": row_dict["direction"],
            "catalyst": row_dict["catalyst"],
            "xgb_win_prob": float(row_dict.get("xgb_win_prob", 0.5)) if row_dict.get("xgb_win_prob") is not None else 0.5,
            "sentinel_verdict": row_dict.get("sentinel_verdict") or "NOT_CHECKED",
            "conviction": row_dict.get("conviction", ""),
            "warning_tag": row_dict.get("warning_tag", "")
        }
        if "z_vol" in row_dict:
            trade_obj["z_vol"] = row_dict["z_vol"]
        if "growth" in row_dict:
            trade_obj["growth"] = row_dict["growth"]
            
        result.append(trade_obj)
    return result

async def get_live_trades() -> list[dict[str, Any]]:
    """Fetch all active live trades."""
    return await asyncio.to_thread(_get_trades_sync, "trade_log")

async def get_archived_trades() -> list[dict[str, Any]]:
    """Fetch all archived trades."""
    return await asyncio.to_thread(_get_trades_sync, "trade_log_archive")

def _get_active_trade_sync(ticker: str) -> dict[str, Any] | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trade_log WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1", (ticker.upper(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        row_dict = dict(row)
        trade_obj = {
            "id": row_dict["id"],
            "timestamp": row_dict["timestamp"],
            "ticker": row_dict["ticker"],
            "price": row_dict["price"],
            "is_whale": bool(row_dict["is_whale"]),
            "win_prob": row_dict["win_prob"],
            "sl": row_dict["sl"],
            "tp": row_dict["tp"],
            "strategy": row_dict["strategy"],
            "direction": row_dict["direction"],
            "catalyst": row_dict["catalyst"],
            "xgb_win_prob": float(row_dict.get("xgb_win_prob", 0.5)) if row_dict.get("xgb_win_prob") is not None else 0.5,
            "sentinel_verdict": row_dict.get("sentinel_verdict") or "NOT_CHECKED",
            "conviction": row_dict.get("conviction", ""),
            "warning_tag": row_dict.get("warning_tag", "")
        }
        if "z_vol" in row_dict:
            trade_obj["z_vol"] = row_dict["z_vol"]
        if "growth" in row_dict:
            trade_obj["growth"] = row_dict["growth"]
        return trade_obj
    return None

async def get_active_trade(ticker: str) -> dict[str, Any] | None:
    """Fetch the latest active trade for a ticker if it exists in the live trade log."""
    return await asyncio.to_thread(_get_active_trade_sync, ticker)

OUTCOME_IN_CATALYST = re.compile(r"\s*\|?\s*Closed\s+(\w+)\s+at\s+([-\d.]+)\s*$")


def _backfill_archive_outcomes(conn) -> int:
    """Recovers exit_price/outcome from rows written before the dedicated columns existed.

    Older builds appended "| Closed {outcome} at {price}" to the free-text catalyst field,
    so a closed trade could not be joined to the prediction that opened it without parsing
    prose. This lifts that text into the real columns and restores the original catalyst.
    Idempotent: rows that already carry an outcome are skipped.
    """
    cursor = conn.cursor()
    try:
        rows = cursor.execute(
            "SELECT id, catalyst FROM trade_log_archive "
            "WHERE outcome IS NULL AND catalyst LIKE '%Closed %'"
        ).fetchall()
    except Exception as e:
        log.warning(f"Archive backfill skipped: {e}")
        return 0

    fixed = 0
    for row in rows:
        rid, catalyst = row["id"], (row["catalyst"] or "")
        m = OUTCOME_IN_CATALYST.search(catalyst)
        if not m:
            continue
        try:
            exit_price = float(m.group(2))
        except ValueError:
            continue
        cursor.execute(
            "UPDATE trade_log_archive SET outcome = ?, exit_price = ?, catalyst = ? WHERE id = ?",
            (m.group(1), exit_price, catalyst[:m.start()].rstrip(" |"), rid),
        )
        fixed += 1

    if fixed:
        conn.commit()
        log.info(f"Backfilled exit_price/outcome on {fixed} archived trade(s) from legacy catalyst text.")
    return fixed


def _close_trade_sync(ticker: str, exit_price: float, outcome: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trade_log WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1", (ticker.upper(),))
    row = cursor.fetchone()
    if row:
        trade_id = row["id"]
        row_dict = dict(row)
        # We don't want to insert the original ID into the archive table
        if "id" in row_dict:
            del row_dict["id"]
            
        # Store the exit in real columns; the catalyst stays the catalyst.
        row_dict["exit_price"] = exit_price
        row_dict["outcome"] = outcome
        
        insert_cols = list(row_dict.keys())
        col_str = ", ".join(insert_cols)
        val_str = ", ".join(["?"] * len(insert_cols))
        
        cursor.execute(f"INSERT INTO trade_log_archive ({col_str}) VALUES ({val_str})", tuple(row_dict.values()))
        cursor.execute("DELETE FROM trade_log WHERE id = ?", (trade_id,))
    conn.commit()
    conn.close()

async def close_trade(ticker: str, exit_price: float, outcome: str):
    """Move an active trade from trade_log to trade_log_archive and record exit price/outcome."""
    await asyncio.to_thread(_close_trade_sync, ticker, exit_price, outcome)

def _clear_table_sync(table: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()

async def clear_live_trades():
    """Clear all trades from the live trade log."""
    await asyncio.to_thread(_clear_table_sync, "trade_log")

async def clear_archived_trades():
    """Clear all trades from the archived trade log."""
    await asyncio.to_thread(_clear_table_sync, "trade_log_archive")
