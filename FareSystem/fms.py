"""
Field Management System (FMS) state and periodic loop.

Responsibilities:
- Hold shared match state (number, duration, running window).
- Own the authoritative lists of teams and fares.
- Provide a thread-safe periodic() loop that:
  - Advances each fare’s state machine (pickup/dropoff/payment).
  - Spawns new fares using fare_gen when below TARGET_FARES, throttled by do_generation().
- Expose small helpers to configure/start/cancel a match.

Threading:
- All access to shared state (fares, teams, match* vars) should be under `mutex`.
- periodic() is intended to run on a background thread.
"""

import time
import random
import sys
from pathlib import Path

# Make the Databases package importable from FareSystem.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Databases"))
try:
    import recorder
    _HAS_RECORDER = True
except ImportError:
    recorder = None
    _HAS_RECORDER = False

from fare_gen import generate_fare
from utils import Point
from fare import Fare
from team import Team
from threading import Lock

from fare_types import FareType

# Match state (protected by mutex)
matchRunning = False          # True while an active match is in progress
matchPaused  = False          # True when match has been paused (timer held)
matchNum = 0                  # Current match number (configurable via config_match)
matchSeed = 0                 # RNG seed for reproducible fare generation (configurable via config_match)
matchDuration = 0             # Match duration in seconds (configurable via config_match)
matchEndTime = 0              # UTC epoch timestamp when the current match ends (0 => not running)
matchTimeRemain = 0           # Seconds remaining at the moment of pause
_pauseStart = 0               # epoch when pause_match() was called (used to shift fare timestamps)

# Active fare list (authoritative). Individual Fare objects manage their own flags.
fares: list[Fare] = []

# Global mutex protecting all shared state above (fares, teams, match variables).
mutex = Lock()

# Lazy way to quickly generate some dummy fares (disabled by default).
# Add Points to this list if you want static seed fares at startup.
points = [
    # Point(0, 2),
    # Point(3, 0),
    # Point(0, -4),
    # Point(-5, 0),
]
# Seed fares use the first configured fare type
default_fare_type = next(iter(FareType))
for point in points:
    fares.append(Fare(Point(0, 0), point, default_fare_type))

# Registered teams participating in the current round, keyed by team number.
# Populated at runtime via the admin UI; starts empty.
teams: dict[int, Team] = {}

# Desired number of concurrently active fares displayed/managed by the system.
TARGET_FARES = 99

# How long (seconds) a team can go without a position update before any fare
# they hold is automatically force-dropped.  This unblocks do_generation() when
# a vehicle leaves the arena without explicitly dropping its fare.
# Must be comfortably longer than normal VPS jitter (VPS publishes at 5 Hz;
# dashboard uses 3 s — we give 15 s to tolerate brief network gaps).
TEAM_ABSENT_TIMEOUT: float = 15.0

# Cooldown timestamp used to stagger fare generation (prevents bursts).
genCooldown = 0

# Fare sequence counter for unique ID generation (resets each match)
fareSequence = 0


def do_generation() -> bool:
    """
    Decide whether to generate a new fare now.

    Returns:
        True if we are below TARGET_FARES and the cooldown has elapsed; False otherwise.
    """
    global fares, genCooldown

    # Count active fares only
    count = 0
    for fare in fares:
        if fare.isActive:
            count += 1

    # Don't over-generate
    if count >= TARGET_FARES:
        return False

    # Enforce cooldown between generations
    if time.time() < genCooldown:
        return False

    # Wait a full 3s to generate the last fare, earlier ones scaled linearly
    genCooldown = time.time() + (count / TARGET_FARES) * 3
    return True


def periodic():
    """
    Main periodic loop.
    - Advances all fares (pickup/dropoff/payment handling).
    - Spawns new fares when allowed by do_generation().
    - Auto-pauses the match when matchEndTime is reached.
    Runs forever; intended to execute on a dedicated background thread.
    Only processes fares while the match is actively running.
    """
    global fares, fareSequence
    while True:
        with mutex:
            if matchRunning:
                # Auto-pause when match time is up
                if matchEndTime > 0 and time.time() >= matchEndTime:
                    print(f"Match {matchNum} time expired — stopping fare system")
                    # Release the lock so pause_match() can acquire it
                    pass  # handled below
                else:
                    # Auto-drop fares held by teams that have left the arena.
                    # A team is considered absent when its lastPosUpdate is older
                    # than TEAM_ABSENT_TIMEOUT *and* it was seen at least once
                    # (lastPosUpdate > 0 — avoids evicting newly-registered teams
                    # that haven't published their first position yet).
                    now = time.time()
                    for team in teams.values():
                        if team.currentFare is None:
                            continue
                        if team.lastPosUpdate <= 0:
                            continue
                        if now - team.lastPosUpdate > TEAM_ABSENT_TIMEOUT:
                            fare_idx = team.currentFare
                            if fare_idx < len(fares):
                                fares[fare_idx].drop_fare(fare_idx, team)
                                print(f"[FMS] Team {team.number} absent for "
                                      f"{now - team.lastPosUpdate:.0f}s — "
                                      f"force-dropped fare {fare_idx}")

                    # Update fare statuses
                    for idx, fare in enumerate(fares):
                        fare.periodic(idx, teams)

                    # Generate a new fare if needed
                    if do_generation():
                        fare = generate_fare(fares, matchNum, fareSequence)
                        if fare is not None:
                            fares.append(fare)
                            fareSequence += 1
                            print(f"New Fare (ID: {fare.unique_id})")
                            if _HAS_RECORDER:
                                from fare_types import get_base_fare, get_distance_fare, get_load_time_multiplier
                                recorder.record_fare_spawn(
                                    fare.unique_id, matchNum, fare.type.name,
                                    fare.src.x, fare.src.y,
                                    fare.dest.x, fare.dest.y,
                                    fare.dist,
                                    get_base_fare(fare.type),
                                    get_distance_fare(fare.type),
                                    fare.compute_fare(),
                                    fare.compute_karma(),
                                    get_load_time_multiplier(fare.type),
                                    time.time(), fare.expiry,
                                )
                                recorder.record_event(fare.unique_id, matchNum, None, "SPAWNED")
                        else:
                            print("Failed faregen")

        # Auto-pause outside the lock to avoid deadlock
        if matchRunning and matchEndTime > 0 and time.time() >= matchEndTime:
            pause_match()

        # 20 ms sleep ~ 50 Hz update rate
        time.sleep(0.02)


def config_match(num: int, duration: int, seed: int | None = None):
    """
    Configure the next match.
    Applies only if no match is currently running.

    Args:
        num: Match number to display.
        duration: Match duration in seconds.
        seed: RNG seed for fare generation. Defaults to num if not provided.
    """
    global matchNum, matchSeed, matchDuration, matchRunning, matchPaused, matchEndTime, matchTimeRemain
    with mutex:
        # Only apply when match is finished or paused (not actively running)
        if not matchRunning:
            matchNum        = num
            matchSeed       = seed if seed is not None else num
            matchDuration   = duration
            matchEndTime    = 0
            matchRunning    = False
            matchPaused     = False
            matchTimeRemain = 0
            if _HAS_RECORDER:
                recorder.record_match(num, matchSeed, duration)


def _shift_fare_timestamps(secs: float):
    """Shift all fare timeout fields forward by `secs` to compensate for a pause.
    Must be called with mutex held."""
    for fare in fares:
        fare.expiry += secs
        if fare._phaseTimeout != -1:
            fare._phaseTimeout += secs


def start_match():
    """
    Start or resume the match.
    - First start: seeds RNG, resets fare counter, starts from matchDuration.
    - Resume after pause: continues from matchTimeRemain, shifting fare timestamps
      so fares don't age during the pause.
    No-op if already running.
    """
    global matchEndTime, matchRunning, matchPaused, matchTimeRemain, fareSequence, _pauseStart
    with mutex:
        if matchRunning:
            return
        if matchPaused and matchTimeRemain > 0:
            # Shift fare timestamps forward by the duration of the pause
            pauseDuration = time.time() - _pauseStart
            _shift_fare_timestamps(pauseDuration)
            matchEndTime = time.time() + matchTimeRemain
            matchRunning = True
            matchPaused  = False
            print(f"Match {matchNum} resumed with {matchTimeRemain:.1f}s remaining (paused {pauseDuration:.1f}s)")
        else:
            # Fresh start
            random.seed(matchSeed)
            fareSequence = 0
            print(f"Match {matchNum} started with seed={matchSeed}")
            matchEndTime    = time.time() + matchDuration
            matchTimeRemain = matchDuration
            matchRunning    = True
            matchPaused     = False
            if _HAS_RECORDER:
                recorder.record_match_start(matchNum, time.time(), teams)


def pause_match():
    """
    Pause the running match, preserving remaining time.
    Records the pause start time so fares can be shifted on resume.
    No-op if not running.
    """
    global matchEndTime, matchRunning, matchPaused, matchTimeRemain, _pauseStart
    with mutex:
        if matchRunning:
            matchTimeRemain = max(0, matchEndTime - time.time())
            matchEndTime    = 0
            matchRunning    = False
            matchPaused     = True
            _pauseStart     = time.time()
            print(f"Match {matchNum} paused with {matchTimeRemain:.1f}s remaining")
            # Record match end when the timer actually expires (not a mid-match pause).
            if matchTimeRemain == 0 and _HAS_RECORDER:
                recorder.record_match_end(matchNum, time.time(), teams)


def cancel_match():
    """
    Stop and reset the match, discarding remaining time and clearing all fares.
    Used by the Reset button.
    """
    global matchEndTime, matchRunning, matchPaused, matchTimeRemain, fares
    with mutex:
        if _HAS_RECORDER and (matchRunning or matchPaused):
            recorder.record_match_end(matchNum, time.time(), teams)
        matchEndTime    = 0
        matchRunning    = False
        matchPaused     = False
        matchTimeRemain = 0
        fares           = []