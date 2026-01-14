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

from fare_gen import generate_fare
from utils import Point
from fare import Fare
from team import Team
from threading import Lock

from fare_types import FareType

# Match state (protected by mutex)
matchRunning = False          # True while an active match is in progress
matchNum = 0                  # Current match number (configurable via config_match)
matchDuration = 0             # Match duration in seconds (configurable via config_match)
matchEndTime = 0              # UTC epoch timestamp when the current match ends (0 => not running)

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

# Registered teams participating in the match, keyed by team number.
# teams: {int: Team} = {
#     3: Team(3),
#     5: Team(5),
#     7: Team(7),
#     10: Team(10),
# }

# Option A: start with no teams
teams: dict[int, Team] = {}

# Option B: only seed in LAB
# teams: dict[int, Team] = {3: Team(3), 5: Team(5), 7: Team(7), 10: Team(10)} if MODE is OperatingMode.LAB else {}

# Desired number of concurrently active fares displayed/managed by the system.
TARGET_FARES = 5

# Cooldown timestamp used to stagger fare generation (prevents bursts).
genCooldown = 0


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
    """
    global fares
    while True:
        with mutex:
            # Update fare statuses
            for idx, fare in enumerate(fares):
                fare.periodic(idx, teams)

            # Generate a new fare if needed
            if do_generation():
                fare = generate_fare(fares)
                if fare is not None:
                    fares.append(fare)
                    print("New Fare")
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
    global matchNum, matchDuration, matchRunning, matchEndTime
    with mutex:
        # Only apply when match is finished
        if matchEndTime < time.time():
            matchNum = num
            matchDuration = duration
            matchEndTime = 0
            matchRunning = False


def start_match():
    """
    Start the configured match.
    Sets matchEndTime = now + matchDuration and marks matchRunning True.
    No-op if already running.
    """
    global matchEndTime, matchRunning
    with mutex:
        if not matchRunning:
            matchEndTime = time.time() + matchDuration
            matchRunning = True


def cancel_match():
    """
    Cancel an in-progress match immediately by clearing matchEndTime.
    Leaves matchRunning True/False unchanged by design.
    """
    global matchEndTime
    with mutex:
        if matchRunning:
            matchEndTime = 0