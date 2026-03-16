import time

from utils import Point

class Team:
    def __init__(self, number : int):
        self.number = number
        self.money = 1000
        self.karma = 20
        self.currentFare : int or None = None
        self.pos = Point(0, 0)
        self.heading = 0.0  # radians, w.r.t. world X axis
        self.lastPosUpdate = 0
        self.lastStatus = 0

    def update_position(self, pos: Point, heading: float = 0.0):
        self.pos = pos
        self.heading = heading
        self.lastPosUpdate = time.time()