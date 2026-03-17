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
TARGET_FARES = 8

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
    Runs forever; intended to execute on a dedicated background thread.
    Only processes fares while the match is actively running.
    """
    global fares, fareSequence
    while True:
        with mutex:
            if matchRunning:
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
                    else:
                        print("Failed faregen")

        # 20 ms sleep ~ 50 Hz update rate
        time.sleep(0.02)


def config_match(num: int, duration: int):
    """
    Configure the next match.
    Applies only if no match is currently running (matchEndTime < now).

    Args:
        num: Match number to display.
        duration: Match duration in seconds.
    """
    global matchNum, matchDuration, matchRunning, matchPaused, matchEndTime, matchTimeRemain
    with mutex:
        # Only apply when match is finished or paused (not actively running)
        if not matchRunning:
            matchNum        = num
            matchDuration   = duration
            matchEndTime    = 0
            matchRunning    = False
            matchPaused     = False
            matchTimeRemain = 0


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
            random.seed(matchNum)
            fareSequence = 0
            print(f"Match {matchNum} started with seed={matchNum}")
            matchEndTime    = time.time() + matchDuration
            matchTimeRemain = matchDuration
            matchRunning    = True
            matchPaused     = False


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


def cancel_match():
    """
    Stop and reset the match, discarding remaining time and clearing all fares.
    Used by the Reset button.
    """
    global matchEndTime, matchRunning, matchPaused, matchTimeRemain, fares
    with mutex:
        matchEndTime    = 0
        matchRunning    = False
        matchPaused     = False
        matchTimeRemain = 0
        fares           = []