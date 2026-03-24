"""
Recompute match_team_summary from raw event data.

Run this against any competition.db that has missing or corrupt
match_team_summary rows.  All intermediate tables (fare_events, fares,
violations, etc.) must already be present.

Usage
-----
    python recompute_summaries.py                     # uses Databases/competition.db
    python recompute_summaries.py /path/to/custom.db
    python recompute_summaries.py --match 3           # only recompute match 3
    python recompute_summaries.py --dry-run           # print what would be written
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults that match Team.__init__ when check_fails == 0
# ---------------------------------------------------------------------------
DEFAULT_MONEY_START = 1000.0
DEFAULT_KARMA_START = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _avg(lst: list) -> "float | None":
    return sum(lst) / len(lst) if lst else None


def _last_money_karma(conn: sqlite3.Connection, match_id: int, team_id: int):
    """Return (money_end, karma_end) from the latest event that has those fields."""
    row = conn.execute(
        """SELECT money_after, karma_after
           FROM fare_events
           WHERE match_id=? AND team_id=?
             AND money_after IS NOT NULL
           ORDER BY ts DESC
           LIMIT 1""",
        (match_id, team_id),
    ).fetchone()
    if row:
        return row["money_after"], row["karma_after"]
    return None, None


def _money_karma_start(conn: sqlite3.Connection, match_id: int, team_id: int):
    """
    Try to recover starting money/karma.

    Strategy:
    1. If a summary row already exists with non-NULL start values, keep them.
    2. Otherwise fall back to the known defaults (1000 money, 30 karma).
    """
    row = conn.execute(
        "SELECT money_start, karma_start FROM match_team_summary"
        " WHERE match_id=? AND team_id=?",
        (match_id, team_id),
    ).fetchone()
    if row and row["money_start"] is not None:
        return row["money_start"], row["karma_start"]
    return DEFAULT_MONEY_START, DEFAULT_KARMA_START


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_team_summary(
    conn: sqlite3.Connection,
    match_id: int,
    team_id: int,
) -> dict:
    """Return a dict of all match_team_summary column values for one team."""

    # -- Fare counts ----------------------------------------------------------
    counts_raw = conn.execute(
        "SELECT event_type, COUNT(*) AS cnt FROM fare_events"
        " WHERE match_id=? AND team_id=? GROUP BY event_type",
        (match_id, team_id),
    ).fetchall()
    counts = {r["event_type"]: r["cnt"] for r in counts_raw}

    fares_completed    = counts.get("DELIVERED",       0)
    fares_dropped      = counts.get("DROPPED",         0)
    fares_expired_held = counts.get("EXPIRED_CLAIMED", 0)

    # -- Phase durations ------------------------------------------------------
    event_rows = conn.execute(
        """SELECT fare_uid, event_type, ts FROM fare_events
           WHERE match_id=? AND team_id=?
           ORDER BY fare_uid, ts""",
        (match_id, team_id),
    ).fetchall()

    fare_evs: dict[int, dict[str, float]] = defaultdict(dict)
    for row in event_rows:
        uid, etype, ts = row["fare_uid"], row["event_type"], row["ts"]
        if etype not in fare_evs[uid]:
            fare_evs[uid][etype] = ts

    times_to_pickup  = []
    times_loading    = []
    times_to_dropoff = []
    times_per_fare   = []

    for evs in fare_evs.values():
        if "CLAIMED" in evs and "AT_PICKUP_ZONE" in evs:
            times_to_pickup.append(evs["AT_PICKUP_ZONE"] - evs["CLAIMED"])
        if "AT_PICKUP_ZONE" in evs and "LOADED" in evs:
            times_loading.append(evs["LOADED"] - evs["AT_PICKUP_ZONE"])
        if "LOADED" in evs and "AT_DROPOFF_ZONE" in evs:
            times_to_dropoff.append(evs["AT_DROPOFF_ZONE"] - evs["LOADED"])
        if "CLAIMED" in evs and "DELIVERED" in evs:
            times_per_fare.append(evs["DELIVERED"] - evs["CLAIMED"])

    # -- Money / karma --------------------------------------------------------
    money_start, karma_start = _money_karma_start(conn, match_id, team_id)
    money_end, karma_end     = _last_money_karma(conn, match_id, team_id)
    money_earned = (
        money_end - money_start
        if money_end is not None and money_start is not None
        else None
    )

    return {
        "match_id":             match_id,
        "team_id":              team_id,
        "fares_completed":      fares_completed,
        "fares_dropped":        fares_dropped,
        "fares_expired_held":   fares_expired_held,
        "avg_time_to_pickup_s": _avg(times_to_pickup),
        "avg_time_loading_s":   _avg(times_loading),
        "avg_time_to_dropoff_s":_avg(times_to_dropoff),
        "avg_time_per_fare_s":  _avg(times_per_fare),
        "money_start":          money_start,
        "money_end":            money_end,
        "money_earned":         money_earned,
        "karma_start":          karma_start,
        "karma_end":            karma_end,
    }


def upsert_summary(conn: sqlite3.Connection, s: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO match_team_summary(
               match_id, team_id,
               fares_completed, fares_dropped, fares_expired_held,
               avg_time_to_pickup_s, avg_time_loading_s,
               avg_time_to_dropoff_s, avg_time_per_fare_s,
               money_start, money_end, money_earned,
               karma_start, karma_end
           ) VALUES (
               :match_id, :team_id,
               :fares_completed, :fares_dropped, :fares_expired_held,
               :avg_time_to_pickup_s, :avg_time_loading_s,
               :avg_time_to_dropoff_s, :avg_time_per_fare_s,
               :money_start, :money_end, :money_earned,
               :karma_start, :karma_end
           )""",
        s,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    default_db = Path(__file__).resolve().parent / "competition.db"

    parser = argparse.ArgumentParser(description="Recompute match_team_summary from raw events.")
    parser.add_argument("db",        nargs="?", default=str(default_db),
                        help=f"Path to competition.db  (default: {default_db})")
    parser.add_argument("--match",   type=int, default=None,
                        help="Only recompute a specific match_id")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rows that would be written without modifying the DB")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Find all (match_id, team_id) pairs with event data.
    if args.match is not None:
        rows = conn.execute(
            "SELECT DISTINCT match_id, team_id FROM fare_events"
            " WHERE match_id=? AND team_id IS NOT NULL"
            " ORDER BY match_id, team_id",
            (args.match,),
        ).fetchall()
        if not rows:
            print(f"No fare_events found for match {args.match}.")
            conn.close()
            return
    else:
        rows = conn.execute(
            "SELECT DISTINCT match_id, team_id FROM fare_events"
            " WHERE team_id IS NOT NULL"
            " ORDER BY match_id, team_id",
        ).fetchall()

    if not rows:
        print("No fare_events found in the database.")
        conn.close()
        return

    print(f"Recomputing summaries for {len(rows)} (match, team) pairs "
          f"{'[DRY RUN]' if args.dry_run else ''}...")

    written = 0
    current_match = None

    with conn:
        for row in rows:
            match_id, team_id = row["match_id"], row["team_id"]
            if match_id != current_match:
                if current_match is not None:
                    print()
                current_match = match_id
                print(f"  Match {match_id}:", end="")

            summary = compute_team_summary(conn, match_id, team_id)

            if args.dry_run:
                print(f"\n    Team {team_id:3d}: "
                      f"delivered={summary['fares_completed']}, "
                      f"dropped={summary['fares_dropped']}, "
                      f"money_end={summary['money_end']}, "
                      f"karma_end={summary['karma_end']}")
            else:
                upsert_summary(conn, summary)
                print(f" {team_id}", end="", flush=True)
                written += 1

    print()
    if not args.dry_run:
        print(f"Done — {written} summary rows written to {db_path}")
    conn.close()


if __name__ == "__main__":
    main()
