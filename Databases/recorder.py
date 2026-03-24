"""
VPFS Competition Database Recorder.

Off-thread SQLite writer for match results.  The public API is fire-and-forget:
every call enqueues a write and returns immediately so the 50 Hz game loop is
never blocked.

Usage
-----
At server startup (router.py):
    import recorder
    recorder.start()            # default: Databases/competition.db
    # or:
    recorder.start("/path/to/custom.db")

At server shutdown (optional — the writer thread is a daemon so it dies with
the process anyway, but calling stop() flushes the queue cleanly):
    recorder.stop()
"""

import sqlite3
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Default DB path — sibling of this file (Databases/competition.db)
# ---------------------------------------------------------------------------
_DEFAULT_DB = Path(__file__).resolve().parent / "competition.db"

# How often at most to record one position sample per team (seconds).
# 0.2 s = 5 Hz → ~150 k rows per 50-team 10-minute match.
POSITION_SAMPLE_INTERVAL: float = 0.2

# Max rows per SQLite transaction.
_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Schema — embedded so the recorder is self-contained.
# Identical to schema.sql; CREATE TABLE IF NOT EXISTS makes it idempotent.
# ---------------------------------------------------------------------------
_SCHEMA = """\
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS matches (
    match_id    INTEGER PRIMARY KEY,
    seed        INTEGER NOT NULL DEFAULT 0,
    duration_s  INTEGER NOT NULL DEFAULT 0,
    started_at  REAL,
    ended_at    REAL
);

CREATE TABLE IF NOT EXISTS teams (
    team_id    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL DEFAULT '',
    group_name TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS fares (
    fare_uid       INTEGER PRIMARY KEY,
    match_id       INTEGER NOT NULL,
    fare_type      TEXT    NOT NULL,
    src_x          REAL    NOT NULL,
    src_y          REAL    NOT NULL,
    dest_x         REAL    NOT NULL,
    dest_y         REAL    NOT NULL,
    distance_cm    REAL    NOT NULL,
    base_fare      REAL    NOT NULL,
    distance_fare  REAL    NOT NULL,
    total_value    REAL    NOT NULL,
    reputation     REAL    NOT NULL,
    load_time_mult REAL    NOT NULL,
    spawned_at     REAL    NOT NULL,
    expires_at     REAL    NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS fare_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    fare_uid    INTEGER NOT NULL,
    match_id    INTEGER NOT NULL,
    team_id     INTEGER,
    event_type  TEXT    NOT NULL,
    ts          REAL    NOT NULL,
    team_x      REAL,
    team_y      REAL,
    money_after REAL,
    karma_after REAL,
    FOREIGN KEY (fare_uid) REFERENCES fares(fare_uid),
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id)  REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_fare_events_match ON fare_events(match_id);
CREATE INDEX IF NOT EXISTS idx_fare_events_team  ON fare_events(team_id, match_id);
CREATE INDEX IF NOT EXISTS idx_fare_events_fare  ON fare_events(fare_uid);

CREATE TABLE IF NOT EXISTS position_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id  INTEGER NOT NULL,
    team_id   INTEGER NOT NULL,
    ts        REAL    NOT NULL,
    x         REAL    NOT NULL,
    y         REAL    NOT NULL,
    heading   REAL    NOT NULL DEFAULT 0,
    has_fare  INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id)  REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_pos_match_team_ts ON position_samples(match_id, team_id, ts);

CREATE TABLE IF NOT EXISTS referee_assignments (
    match_id     INTEGER NOT NULL,
    team_id      INTEGER NOT NULL,
    referee_id   TEXT    NOT NULL,
    assigned_at  REAL    NOT NULL,
    PRIMARY KEY (match_id, team_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id)  REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS violations (
    violation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id       INTEGER NOT NULL,
    team_id        INTEGER NOT NULL,
    fare_uid       INTEGER,
    violation_type TEXT    NOT NULL,
    referee_id     TEXT    NOT NULL,
    ts             REAL    NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id)  REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_violations_team ON violations(match_id, team_id);
CREATE INDEX IF NOT EXISTS idx_violations_fare ON violations(fare_uid);

CREATE TABLE IF NOT EXISTS achievements (
    achievement_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id         INTEGER NOT NULL,
    team_id          INTEGER NOT NULL,
    fare_uid         INTEGER,
    achievement_type TEXT    NOT NULL,
    granted_by       TEXT    NOT NULL,
    ts               REAL    NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id)  REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_achievements_team ON achievements(match_id, team_id);

CREATE TABLE IF NOT EXISTS match_team_summary (
    match_id              INTEGER NOT NULL,
    team_id               INTEGER NOT NULL,
    fares_completed       INTEGER,
    fares_dropped         INTEGER,
    fares_expired_held    INTEGER,
    avg_time_to_pickup_s  REAL,
    avg_time_loading_s    REAL,
    avg_time_to_dropoff_s REAL,
    avg_time_per_fare_s   REAL,
    money_start           REAL,
    money_end             REAL,
    money_earned          REAL,
    karma_start           REAL,
    karma_end             REAL,
    PRIMARY KEY (match_id, team_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id)  REFERENCES teams(team_id)
);
"""

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_write_queue: queue.Queue = queue.Queue()
_last_pos_ts: dict[int, float] = {}            # team_id -> last sample timestamp
_match_start_money: dict[int, float] = {}      # team_id -> money at match start
_match_start_karma: dict[int, float] = {}      # team_id -> karma at match start
_running = False
_db_path: Path = _DEFAULT_DB
_writer_thread: threading.Thread | None = None

# Lock protecting all synchronous (direct-connection) DB writes so they don't
# race with each other.  The async writer thread has its own connection and is
# serialized by SQLite's WAL writer lock.
_sync_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Writer thread
# ---------------------------------------------------------------------------
def _writer() -> None:
    conn = sqlite3.connect(str(_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    while True:
        # Block until at least one item arrives.
        item = _write_queue.get()
        if item is None:          # shutdown sentinel
            break

        # Non-blocking drain to build a batch.
        batch = [item]
        try:
            while len(batch) < _BATCH_SIZE:
                nxt = _write_queue.get_nowait()
                if nxt is None:   # sentinel inside batch — stop after this flush
                    batch.append(nxt)
                    break
                batch.append(nxt)
        except queue.Empty:
            pass

        # Write the entire batch in one transaction.
        stop_after = False
        try:
            with conn:
                for entry in batch:
                    if entry is None:
                        stop_after = True
                        break
                    sql, params = entry
                    conn.execute(sql, params)
        except Exception as exc:
            print(f"[recorder] write error: {exc}")

        if stop_after:
            break

    conn.close()


# Whether database recording is active.  Can be toggled at runtime via
# set_recording_enabled() without stopping/starting the writer thread.
_recording_enabled: bool = True


def set_recording_enabled(enabled: bool) -> None:
    """Enable or disable database recording at runtime."""
    global _recording_enabled
    _recording_enabled = bool(enabled)
    state = "enabled" if _recording_enabled else "disabled"
    print(f"[recorder] recording {state}")


def is_recording_enabled() -> bool:
    """Return current recording state."""
    return _recording_enabled


def _enqueue(sql: str, params: tuple) -> None:
    if _running and _recording_enabled:
        _write_queue.put((sql, params))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def start(db_path: "Path | str | None" = None) -> None:
    """Initialise schema and start the background writer thread.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _running, _db_path, _writer_thread

    if _running:
        return

    if db_path is not None:
        _db_path = Path(db_path)

    # Apply schema before the writer thread connects (idempotent).
    conn = sqlite3.connect(str(_db_path))
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()

    _running = True
    _writer_thread = threading.Thread(
        target=_writer, daemon=True, name="db-recorder"
    )
    _writer_thread.start()
    print(f"[recorder] started — DB: {_db_path}")


def stop() -> None:
    """Flush pending writes and stop the writer thread gracefully."""
    global _running
    if not _running:
        return
    _running = False
    _write_queue.put(None)     # sentinel
    if _writer_thread:
        _writer_thread.join(timeout=10)
    print("[recorder] stopped")


# ---------------------------------------------------------------------------
# Match recording
# ---------------------------------------------------------------------------
def record_match(match_id: int, seed: int, duration_s: int) -> None:
    """Call when a match is configured (fms.config_match).

    Uses ON CONFLICT DO UPDATE so that re-configuring an existing match number
    (e.g. after a Reset) only updates seed/duration and never wipes the
    started_at / ended_at timestamps that were already recorded.
    """
    _enqueue(
        """
        INSERT INTO matches(match_id, seed, duration_s) VALUES (?,?,?)
        ON CONFLICT(match_id) DO UPDATE SET
            seed=excluded.seed,
            duration_s=excluded.duration_s
        """,
        (match_id, seed, duration_s),
    )


def record_match_start(match_id: int, started_at: float, teams: dict) -> None:
    """Call when a match starts (fms.start_match).

    ``teams`` is the fms.teams dict {team_id: Team}.
    Snapshots starting money/karma for each team and upserts team registry rows.
    """
    _enqueue(
        "UPDATE matches SET started_at=? WHERE match_id=?",
        (started_at, match_id),
    )

    _match_start_money.clear()
    _match_start_karma.clear()
    for tid, t in teams.items():
        _match_start_money[tid] = t.money
        _match_start_karma[tid] = t.karma
        name = getattr(t, "name", f"Team {tid}")
        # e.g. "Aylesbury 3" → group "Aylesbury"
        group = name.rsplit(" ", 1)[0] if " " in name else name
        _enqueue(
            "INSERT OR IGNORE INTO teams(team_id, name, group_name) VALUES (?,?,?)",
            (tid, name, group),
        )


def record_match_end(match_id: int, ended_at: float, teams: dict) -> None:
    """Call when a match ends (timer expiry or cancel).

    Writes the ended_at timestamp and kicks off asynchronous summary computation.
    """
    _enqueue(
        "UPDATE matches SET ended_at=? WHERE match_id=?",
        (ended_at, match_id),
    )
    # Snapshot final state; compute summaries in a separate thread so the
    # writer queue has time to flush all remaining events first.
    end_snapshot = {tid: (t.money, t.karma) for tid, t in teams.items()}
    # Snapshot start money/karma NOW (before any new match could clear the
    # module-level dicts in record_match_start) so the summary thread has a
    # stable copy even if a new match begins within the 2-second sleep window.
    start_money = dict(_match_start_money)
    start_karma = dict(_match_start_karma)
    threading.Thread(
        target=_compute_summaries,
        args=(match_id, end_snapshot, start_money, start_karma),
        daemon=True,
        name="db-summaries",
    ).start()


# ---------------------------------------------------------------------------
# Fare recording
# ---------------------------------------------------------------------------
def record_fare_spawn(
    fare_uid: int,
    match_id: int,
    fare_type: str,
    src_x: float,
    src_y: float,
    dest_x: float,
    dest_y: float,
    distance_cm: float,
    base_fare: float,
    distance_fare: float,
    total_value: float,
    reputation: float,
    load_time_mult: float,
    spawned_at: float,
    expires_at: float,
) -> None:
    """Insert a fare definition row. Called once per fare at spawn time."""
    _enqueue(
        """INSERT OR IGNORE INTO fares(
               fare_uid, match_id, fare_type,
               src_x, src_y, dest_x, dest_y,
               distance_cm, base_fare, distance_fare,
               total_value, reputation, load_time_mult,
               spawned_at, expires_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            fare_uid, match_id, fare_type,
            src_x, src_y, dest_x, dest_y,
            distance_cm, base_fare, distance_fare,
            total_value, reputation, load_time_mult,
            spawned_at, expires_at,
        ),
    )


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------
def record_event(
    fare_uid: int,
    match_id: int,
    team_id: "int | None",
    event_type: str,
    team_x: "float | None" = None,
    team_y: "float | None" = None,
    money_after: "float | None" = None,
    karma_after: "float | None" = None,
) -> None:
    """Append one fare_events row. All callers are fire-and-forget."""
    _enqueue(
        """INSERT INTO fare_events(
               fare_uid, match_id, team_id, event_type, ts,
               team_x, team_y, money_after, karma_after
           ) VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            fare_uid, match_id, team_id, event_type, time.time(),
            team_x, team_y, money_after, karma_after,
        ),
    )


# ---------------------------------------------------------------------------
# Position recording
# ---------------------------------------------------------------------------
def record_position(
    match_id: int,
    team_id: int,
    x: float,
    y: float,
    heading: float,
    has_fare: bool,
) -> None:
    """Record one position sample, throttled to POSITION_SAMPLE_INTERVAL per team."""
    now = time.time()
    if now - _last_pos_ts.get(team_id, 0.0) < POSITION_SAMPLE_INTERVAL:
        return
    _last_pos_ts[team_id] = now
    _enqueue(
        """INSERT INTO position_samples(match_id, team_id, ts, x, y, heading, has_fare)
           VALUES (?,?,?,?,?,?,?)""",
        (match_id, team_id, now, x, y, heading, 1 if has_fare else 0),
    )


# ---------------------------------------------------------------------------
# Summary computation  (runs in a background thread ~2 s after match end)
# ---------------------------------------------------------------------------
def _compute_summaries(
    match_id: int,
    end_snapshot: dict,
    start_money: dict,
    start_karma: dict,
) -> None:
    time.sleep(2)   # give the writer thread time to flush remaining events
    try:
        conn = sqlite3.connect(str(_db_path))
        conn.row_factory = sqlite3.Row

        team_ids = [
            r["team_id"]
            for r in conn.execute(
                "SELECT DISTINCT team_id FROM fare_events "
                "WHERE match_id=? AND team_id IS NOT NULL",
                (match_id,),
            ).fetchall()
        ]

        for team_id in team_ids:
            _write_team_summary(conn, match_id, team_id, end_snapshot, start_money, start_karma)

        # Note: auto-achievements (SAFETY_FIRST, HOLDING_OUT) are now recorded
        # live inside fare.pay_fare() — no batch computation needed here.
        conn.close()
        print(f"[recorder] summaries written for match {match_id} ({len(team_ids)} teams)")
    except Exception as exc:
        print(f"[recorder] summary computation error: {exc}")


def _write_team_summary(
    conn: sqlite3.Connection,
    match_id: int,
    team_id: int,
    end_snapshot: dict,
    start_money: dict,
    start_karma: dict,
) -> None:
    # -- Fare counts ----------------------------------------------------------
    counts_raw = conn.execute(
        "SELECT event_type, COUNT(*) AS cnt FROM fare_events "
        "WHERE match_id=? AND team_id=? GROUP BY event_type",
        (match_id, team_id),
    ).fetchall()
    counts = {r["event_type"]: r["cnt"] for r in counts_raw}

    fares_completed    = counts.get("DELIVERED", 0)
    fares_dropped      = counts.get("DROPPED",   0)
    fares_expired_held = counts.get("EXPIRED_CLAIMED", 0)

    # -- Phase durations (only for fares that reached each milestone) ---------
    event_rows = conn.execute(
        """SELECT fare_uid, event_type, ts FROM fare_events
           WHERE match_id=? AND team_id=?
           ORDER BY fare_uid, ts""",
        (match_id, team_id),
    ).fetchall()

    # Keep the earliest timestamp for each event_type per fare.
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

    def _avg(lst: list) -> "float | None":
        return sum(lst) / len(lst) if lst else None

    # -- Money / karma --------------------------------------------------------
    money_start = start_money.get(team_id)
    karma_start = start_karma.get(team_id)
    money_end, karma_end = end_snapshot.get(team_id, (None, None))
    money_earned = (
        money_end - money_start
        if money_end is not None and money_start is not None
        else None
    )

    conn.execute(
        """INSERT OR REPLACE INTO match_team_summary(
               match_id, team_id,
               fares_completed, fares_dropped, fares_expired_held,
               avg_time_to_pickup_s, avg_time_loading_s,
               avg_time_to_dropoff_s, avg_time_per_fare_s,
               money_start, money_end, money_earned,
               karma_start, karma_end
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            match_id, team_id,
            fares_completed, fares_dropped, fares_expired_held,
            _avg(times_to_pickup), _avg(times_loading),
            _avg(times_to_dropoff), _avg(times_per_fare),
            money_start, money_end, money_earned,
            karma_start, karma_end,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Synchronous helpers  (low-frequency referee reads/deletes)
# ---------------------------------------------------------------------------
def _sync_read(sql: str, params: tuple = ()) -> list:
    """Open a short-lived read connection and return all rows."""
    conn = sqlite3.connect(str(_db_path))
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _sync_write(sql: str, params: tuple = ()) -> int:
    """Open a short-lived write connection (serialised by _sync_lock). Returns rowcount."""
    with _sync_lock:
        conn = sqlite3.connect(str(_db_path))
        try:
            with conn:
                cur = conn.execute(sql, params)
                return cur.rowcount
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Referee assignment recording
# ---------------------------------------------------------------------------
def record_assignment(match_id: int, team_id: int, referee_id: str) -> None:
    """Upsert the referee assignment for a team in the current match.

    Also clears any previous team assignment for this referee so each referee
    can only watch one team at a time.
    """
    # Synchronously drop old assignment for this referee before enqueuing new one.
    with _sync_lock:
        conn = sqlite3.connect(str(_db_path))
        try:
            with conn:
                conn.execute(
                    "DELETE FROM referee_assignments WHERE match_id=? AND referee_id=?",
                    (match_id, referee_id),
                )
        finally:
            conn.close()
    _enqueue(
        "INSERT OR REPLACE INTO referee_assignments(match_id, team_id, referee_id, assigned_at)"
        " VALUES (?,?,?,?)",
        (match_id, team_id, referee_id, time.time()),
    )


def clear_referee_assignments(match_id: int, referee_id: str) -> None:
    """Remove all team assignments for a referee in this match (used on unassign)."""
    _sync_write(
        "DELETE FROM referee_assignments WHERE match_id=? AND referee_id=?",
        (match_id, referee_id),
    )


def get_assignments(match_id: int) -> dict:
    """Return {team_id: {referee_id, assigned_at}} for all assignments in *match_id*."""
    rows = _sync_read(
        "SELECT team_id, referee_id FROM referee_assignments WHERE match_id=?",
        (match_id,),
    )
    return {r["team_id"]: r["referee_id"] for r in rows}


# ---------------------------------------------------------------------------
# Violation recording
# ---------------------------------------------------------------------------
def record_violation(
    match_id: int,
    team_id: int,
    fare_uid: "int | None",
    violation_type: str,
    referee_id: str,
) -> None:
    """Async insert of one violation row."""
    _enqueue(
        "INSERT INTO violations(match_id, team_id, fare_uid, violation_type, referee_id, ts)"
        " VALUES (?,?,?,?,?,?)",
        (match_id, team_id, fare_uid, violation_type, referee_id, time.time()),
    )


def delete_latest_violation(match_id: int, team_id: int, violation_type: str) -> bool:
    """Delete the most recent violation of *violation_type* for this team. Returns True if one was deleted."""
    rows = _sync_read(
        "SELECT violation_id FROM violations"
        " WHERE match_id=? AND team_id=? AND violation_type=?"
        " ORDER BY ts DESC LIMIT 1",
        (match_id, team_id, violation_type),
    )
    if not rows:
        return False
    _sync_write("DELETE FROM violations WHERE violation_id=?", (rows[0]["violation_id"],))
    return True


# ---------------------------------------------------------------------------
# Achievement recording
# ---------------------------------------------------------------------------
def record_achievement(
    match_id: int,
    team_id: int,
    fare_uid: "int | None",
    achievement_type: str,
    granted_by: str,
) -> None:
    """Async insert of one achievement row."""
    _enqueue(
        "INSERT INTO achievements(match_id, team_id, fare_uid, achievement_type, granted_by, ts)"
        " VALUES (?,?,?,?,?,?)",
        (match_id, team_id, fare_uid, achievement_type, granted_by, time.time()),
    )


def delete_latest_achievement(match_id: int, team_id: int, achievement_type: str) -> bool:
    """Delete the most recent achievement of *achievement_type* for this team. Returns True if one was deleted."""
    rows = _sync_read(
        "SELECT achievement_id FROM achievements"
        " WHERE match_id=? AND team_id=? AND achievement_type=?"
        " ORDER BY ts DESC LIMIT 1",
        (match_id, team_id, achievement_type),
    )
    if not rows:
        return False
    _sync_write("DELETE FROM achievements WHERE achievement_id=?", (rows[0]["achievement_id"],))
    return True


# ---------------------------------------------------------------------------
# Live team counts  (used by the referee panel)
# ---------------------------------------------------------------------------
def get_team_counts(match_id: int, team_id: int) -> dict:
    """Return violation and achievement counts for *team_id* in *match_id*.

    Returns::

        {
            "violations":   {"STANDARD": int, "SEVERE": int},
            "achievements": {"ZERO_DUCKS_GIVEN": int, "YOU_SPIN_ME_ROUND": int,
                             "SAFETY_FIRST": int, "HOLDING_OUT": int},
        }
    """
    vrows = _sync_read(
        "SELECT violation_type, COUNT(*) AS cnt FROM violations"
        " WHERE match_id=? AND team_id=? GROUP BY violation_type",
        (match_id, team_id),
    )
    arows = _sync_read(
        "SELECT achievement_type, COUNT(*) AS cnt FROM achievements"
        " WHERE match_id=? AND team_id=? GROUP BY achievement_type",
        (match_id, team_id),
    )
    return {
        "violations":   {r["violation_type"]: r["cnt"] for r in vrows},
        "achievements": {r["achievement_type"]: r["cnt"] for r in arows},
    }


# ---------------------------------------------------------------------------
# Auto-achievement computation  (called at match end from _compute_summaries)
# ---------------------------------------------------------------------------
def _compute_auto_achievements(match_id: int, conn: sqlite3.Connection) -> None:
    """Grant SAFETY_FIRST and HOLDING_OUT awards for all completed fares in *match_id*.

    Must be called with an open, row_factory-enabled connection (already inside
    _compute_summaries which opens its own connection).
    """
    delivered = conn.execute(
        """SELECT fe.fare_uid, fe.team_id, f.fare_type
           FROM fare_events fe
           JOIN fares f USING(fare_uid)
           WHERE fe.match_id=? AND fe.event_type='DELIVERED'""",
        (match_id,),
    ).fetchall()

    now = time.time()
    for row in delivered:
        fare_uid  = row["fare_uid"]
        team_id   = row["team_id"]
        fare_type = row["fare_type"]

        # --- Safety First: no violations at all during this fare ---------------
        viol_count = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE fare_uid=? AND team_id=?",
            (fare_uid, team_id),
        ).fetchone()[0]
        already = conn.execute(
            "SELECT COUNT(*) FROM achievements"
            " WHERE match_id=? AND team_id=? AND fare_uid=? AND achievement_type='SAFETY_FIRST'",
            (match_id, team_id, fare_uid),
        ).fetchone()[0]
        if viol_count == 0 and not already:
            conn.execute(
                "INSERT INTO achievements(match_id, team_id, fare_uid, achievement_type, granted_by, ts)"
                " VALUES (?,?,?,?,?,?)",
                (match_id, team_id, fare_uid, "SAFETY_FIRST", "SYSTEM", now),
            )

        # --- Holding Out for a Hero: completed a SPECIAL fare ------------------
        if fare_type == "SPECIAL":
            already = conn.execute(
                "SELECT COUNT(*) FROM achievements"
                " WHERE match_id=? AND team_id=? AND fare_uid=? AND achievement_type='HOLDING_OUT'",
                (match_id, team_id, fare_uid),
            ).fetchone()[0]
            if not already:
                conn.execute(
                    "INSERT INTO achievements(match_id, team_id, fare_uid, achievement_type, granted_by, ts)"
                    " VALUES (?,?,?,?,?,?)",
                    (match_id, team_id, fare_uid, "HOLDING_OUT", "SYSTEM", now),
                )

    conn.commit()

