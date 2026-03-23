import time

from utils import Point

class Team:
    def __init__(self, number: int, check_fails: int = 0):
        self.number = number
        self.money = 1000
        self.karma = max(-100, min(100, 30 - check_fails * 2))
        self.currentFare : int or None = None
        self.pos = Point(0, 0)
        self.heading = 0.0  # radians, w.r.t. world X axis
        self.lastPosUpdate = 0
        self.lastStatus = 0

        # Match-level violation totals (used by referee panel; persist across fares)
        self.standard_violations: int = 0
        self.severe_violations:   int = 0

    def update_position(self, pos: Point, heading: float = 0.0):
        self.pos = pos
        self.heading = heading
        self.lastPosUpdate = time.time()

    @property
    def money(self):
        return self._money

    @money.setter
    def money(self, value):
        self._money = max(0, value)