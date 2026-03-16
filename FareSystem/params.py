from typing import Final
from enum import Enum

class OperatingMode(str, Enum):
    LAB = "Lab"
    HOME = "Home"
    MATCH = "Match"

MODE = OperatingMode.MATCH

POSITION_TOLERANCE = 15
PICKUP_DURATION = 5

BASE_FARE: Final[float] = 10.0
DIST_FARE_NORMAL: Final[float] = 10.0
DIST_FARE_SUBSIDIZED: Final[float] = 5.0

REPUTATION_NORMAL: Final[int] = 5
REPUTATION_SUBSIDIZED: Final[int] = 10