#!/usr/bin/env python3
"""
VPFS Pre-competition database initializer.

Run this script once before each competition (or each match day) to create
a fresh SQLite database with all required tables.

Usage
-----
    python init_db.py                          # creates competition.db alongside this script
    python init_db.py /path/to/results.db      # custom location

If the file already exists, existing tables are preserved (all CREATE statements
use IF NOT EXISTS), so re-running is safe and never loses data.
"""

import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_FILE = SCRIPT_DIR / "schema.sql"
DEFAULT_DB  = SCRIPT_DIR / "competition.db"


def init_db(db_path: Path) -> None:
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")

    schema = SCHEMA_FILE.read_text(encoding="utf-8")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(schema)
        conn.commit()
        print(f"Database initialised: {db_path}")
        _print_table_info(conn)
    finally:
        conn.close()


def _print_table_info(conn: sqlite3.Connection) -> None:
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print(f"  Tables created/verified ({len(tables)}):")
    for (name,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
        print(f"    {name:<30} {count} rows")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    try:
        init_db(target)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
