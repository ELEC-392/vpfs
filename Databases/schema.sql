-- VPFS Competition Database Schema
-- Apply with: python init_db.py [path/to/competition.db]
-- All timestamps are Unix epoch seconds (REAL) from Python's time.time().

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

-- ---------------------------------------------------------------------------
-- Matches
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY,   -- same as fms.matchNum
    seed            INTEGER NOT NULL DEFAULT 0,
    duration_s      INTEGER NOT NULL DEFAULT 0,
    started_at      REAL,                  -- NULL until match actually starts
    ended_at        REAL                   -- NULL until match finishes
);

-- ---------------------------------------------------------------------------
-- Team registry  (populated at match-start from teams.yaml)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    team_id         INTEGER PRIMARY KEY,   -- kit number (team.number)
    name            TEXT    NOT NULL DEFAULT '',
    group_name      TEXT    NOT NULL DEFAULT ''  -- 'Aylesbury', 'Blekinge', …
);

-- ---------------------------------------------------------------------------
-- Fares  (one row per fare, written at spawn time)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fares (
    fare_uid        INTEGER PRIMARY KEY,   -- fare.unique_id = match_id*1000 + seq
    match_id        INTEGER NOT NULL,
    fare_type       TEXT    NOT NULL,      -- 'STANDARD', 'SPECIAL', …
    src_x           REAL    NOT NULL,
    src_y           REAL    NOT NULL,
    dest_x          REAL    NOT NULL,
    dest_y          REAL    NOT NULL,
    distance_cm     REAL    NOT NULL,
    base_fare       REAL    NOT NULL,
    distance_fare   REAL    NOT NULL,
    total_value     REAL    NOT NULL,
    reputation      REAL    NOT NULL,
    load_time_mult  REAL    NOT NULL,
    spawned_at      REAL    NOT NULL,
    expires_at      REAL    NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

-- ---------------------------------------------------------------------------
-- Fare events  (the core audit log — one row per state transition)
--
-- event_type values:
--   SPAWNED          fare created
--   CLAIMED          team claimed the fare
--   AT_PICKUP_ZONE   team entered the pickup zone (may repeat if team leaves/re-enters)
--   LOADED           passenger picked up (loading timer elapsed)
--   AT_DROPOFF_ZONE  team entered the dropoff zone
--   DELIVERED        fare completed
--   PAID             money credited to team
--   DROPPED          team voluntarily dropped the fare
--   EXPIRED          fare expired with no team holding it
--   EXPIRED_CLAIMED  fare expired while a team was holding it
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fare_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    fare_uid        INTEGER NOT NULL,
    match_id        INTEGER NOT NULL,
    team_id         INTEGER,               -- NULL for SPAWNED / EXPIRED
    event_type      TEXT    NOT NULL,
    ts              REAL    NOT NULL,
    team_x          REAL,                  -- team position at event moment
    team_y          REAL,
    money_after     REAL,                  -- snapshot of team.money (for PAID / DROPPED)
    karma_after     REAL,
    FOREIGN KEY (fare_uid)  REFERENCES fares(fare_uid),
    FOREIGN KEY (match_id)  REFERENCES matches(match_id),
    FOREIGN KEY (team_id)   REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_fare_events_match  ON fare_events(match_id);
CREATE INDEX IF NOT EXISTS idx_fare_events_team   ON fare_events(team_id, match_id);
CREATE INDEX IF NOT EXISTS idx_fare_events_fare   ON fare_events(fare_uid);

-- ---------------------------------------------------------------------------
-- Position samples  (replay + distance computation)
-- Written at ~5 Hz per team; has_fare=1 when team held a fare at sample time.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS position_samples (
    sample_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id    INTEGER NOT NULL,
    team_id     INTEGER NOT NULL,
    ts          REAL    NOT NULL,
    x           REAL    NOT NULL,
    y           REAL    NOT NULL,
    heading     REAL    NOT NULL DEFAULT 0,
    has_fare    INTEGER NOT NULL DEFAULT 0,  -- 1 = carrying a fare
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id)  REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_pos_match_team_ts ON position_samples(match_id, team_id, ts);

-- ---------------------------------------------------------------------------
-- Referee assignments  (which referee is watching which team per match)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS referee_assignments (
    match_id     INTEGER NOT NULL,
    team_id      INTEGER NOT NULL,
    referee_id   TEXT    NOT NULL,
    assigned_at  REAL    NOT NULL,
    PRIMARY KEY (match_id, team_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id)  REFERENCES teams(team_id)
);

-- ---------------------------------------------------------------------------
-- Safety violations  (one row per violation event)
--
-- violation_type values: 'STANDARD', 'SEVERE'
-- fare_uid is the fare active at the time of the violation (NULL if no fare).
-- Linking to fare_uid enables automatic "Safety First" computation at match end:
--   a DELIVERED fare with zero violations for that fare_uid + team earns the award.
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- Achievements  (one row per award — both manual and auto-computed)
--
-- achievement_type values:
--   SAFETY_FIRST      auto-granted at match end (DELIVERED fare with no violations)
--   HOLDING_OUT       auto-granted at match end (SPECIAL fare delivered)
--   ZERO_DUCKS_GIVEN  manually granted by referee (risky legal maneuver pays off)
--   YOU_SPIN_ME_ROUND manually granted by referee (roundabout entry/exit clean)
-- granted_by: referee_id for manual grants, 'SYSTEM' for auto-computed grants.
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- Per-match per-team summary  (written at match end by the recorder)
-- All times in seconds.
--
-- Useful query — teams sorted by fares completed:
--   SELECT t.name, s.fares_completed, s.money_end
--   FROM match_team_summary s JOIN teams t USING(team_id)
--   WHERE s.match_id=? ORDER BY s.fares_completed DESC
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_team_summary (
    match_id                INTEGER NOT NULL,
    team_id                 INTEGER NOT NULL,

    fares_completed         INTEGER,
    fares_dropped           INTEGER,
    fares_expired_held      INTEGER,

    -- Average phase durations across all completed fares
    avg_time_to_pickup_s    REAL,   -- CLAIMED → AT_PICKUP_ZONE
    avg_time_loading_s      REAL,   -- AT_PICKUP_ZONE → LOADED
    avg_time_to_dropoff_s   REAL,   -- LOADED → AT_DROPOFF_ZONE
    avg_time_per_fare_s     REAL,   -- CLAIMED → DELIVERED

    money_start             REAL,
    money_end               REAL,
    money_earned            REAL,
    karma_start             REAL,
    karma_end               REAL,

    PRIMARY KEY (match_id, team_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id)  REFERENCES teams(team_id)
);
