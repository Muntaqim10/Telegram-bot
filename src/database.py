import asyncio
import csv
import os
import sqlite3
from datetime import datetime
from typing import Any

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
            print(f"[DB] Error migrating live CSV: {e}")

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
            print(f"[DB] Error migrating archived CSV: {e}")
            
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
            z_vol TEXT NOT NULL,
            is_whale INTEGER NOT NULL,
            growth TEXT NOT NULL,
            win_prob TEXT NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL,
            catalyst TEXT NOT NULL,
            xgb_win_prob REAL DEFAULT 0.5,
            sentinel_verdict TEXT DEFAULT 'NOT_CHECKED'
        )
    """)
    
    # Create archived trade log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_log_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            z_vol TEXT NOT NULL,
            is_whale INTEGER NOT NULL,
            growth TEXT NOT NULL,
            win_prob TEXT NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL,
            catalyst TEXT NOT NULL,
            xgb_win_prob REAL DEFAULT 0.5,
            sentinel_verdict TEXT DEFAULT 'NOT_CHECKED'
        )
    """)
    conn.commit()
    
    # Migrate existing tables to add sentinel columns if they don't exist
    for table in ['trade_log', 'trade_log_archive']:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN xgb_win_prob REAL DEFAULT 0.5")
        except Exception:
            pass  # Column already exists
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN sentinel_verdict TEXT DEFAULT 'NOT_CHECKED'")
        except Exception:
            pass  # Column already exists
    conn.commit()
    
    # Run CSV migrations
    _migrate_csv_data(conn)
    conn.close()

async def init_db():
    """Asynchronously initialize the database."""
    await asyncio.to_thread(_init_db_sync)

def _add_trade_sync(table: str, trade: dict[str, Any]):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        INSERT INTO {table} (
            timestamp, ticker, price, z_vol, is_whale, growth, win_prob, sl, tp, strategy, direction, catalyst,
            xgb_win_prob, sentinel_verdict
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        trade.get("ticker", "").upper(),
        float(trade.get("price", 0.0)),
        str(trade.get("z_vol", "0.0σ")),
        1 if trade.get("is_whale") else 0,
        str(trade.get("growth", "0%")),
        str(trade.get("win_prob", "50%")),
        float(trade.get("sl", 0.0)),
        float(trade.get("tp", 0.0)),
        trade.get("strategy", "Unknown"),
        trade.get("direction", "Long"),
        trade.get("catalyst", ""),
        float(trade.get("xgb_win_prob", 0.5)),
        trade.get("sentinel_verdict", "NOT_CHECKED")
    ))
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
        result.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "ticker": r["ticker"],
            "price": r["price"],
            "z_vol": r["z_vol"],
            "is_whale": bool(r["is_whale"]),
            "growth": r["growth"],
            "win_prob": r["win_prob"],
            "sl": r["sl"],
            "tp": r["tp"],
            "strategy": r["strategy"],
            "direction": r["direction"],
            "catalyst": r["catalyst"],
            "xgb_win_prob": float(r["xgb_win_prob"]) if r["xgb_win_prob"] is not None else 0.5,
            "sentinel_verdict": r["sentinel_verdict"] or "NOT_CHECKED"
        })
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
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "ticker": row["ticker"],
            "price": row["price"],
            "z_vol": row["z_vol"],
            "is_whale": bool(row["is_whale"]),
            "growth": row["growth"],
            "win_prob": row["win_prob"],
            "sl": row["sl"],
            "tp": row["tp"],
            "strategy": row["strategy"],
            "direction": row["direction"],
            "catalyst": row["catalyst"],
            "xgb_win_prob": float(row["xgb_win_prob"]) if row["xgb_win_prob"] is not None else 0.5,
            "sentinel_verdict": row["sentinel_verdict"] or "NOT_CHECKED"
        }
    return None

async def get_active_trade(ticker: str) -> dict[str, Any] | None:
    """Fetch the latest active trade for a ticker if it exists in the live trade log."""
    return await asyncio.to_thread(_get_active_trade_sync, ticker)

def _close_trade_sync(ticker: str, exit_price: float, outcome: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trade_log WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1", (ticker.upper(),))
    row = cursor.fetchone()
    if row:
        trade_id = row["id"]
        new_catalyst = f"{row['catalyst']} | Closed {outcome} at {exit_price}" if row['catalyst'] else f"Closed {outcome} at {exit_price}"
        cursor.execute("""
            INSERT INTO trade_log_archive (
                timestamp, ticker, price, z_vol, is_whale, growth, win_prob, sl, tp, strategy, direction, catalyst,
                xgb_win_prob, sentinel_verdict
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["timestamp"], row["ticker"], row["price"], row["z_vol"], row["is_whale"],
            row["growth"], row["win_prob"], row["sl"], row["tp"], row["strategy"],
            row["direction"], new_catalyst, row["xgb_win_prob"], row["sentinel_verdict"]
        ))
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

# Run database table initialization immediately on import
_init_db_sync()
