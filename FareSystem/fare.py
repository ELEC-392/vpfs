import random

from utils import Point
from team import Team
import time
from params import POSITION_TOLERANCE, PICKUP_DURATION
from fare_types import (
    FareType,
    get_base_fare,
    get_distance_fare,
    get_load_time_multiplier,
    get_reputation,
)

class Fare:
    def __init__(self, src : Point, dest: Point, fare_type: FareType):
        """
        :param src: Location the ducky is picked up at
        :param dest: Location the ducky is delivered to
        """
        self.src = src
        self.dest = dest
        self.dist = src.dist(dest)
        self.type = fare_type
        self.expiry = time.time() + random.randint(60, 150)
        self.team : int | None = None
        # Timeout used to create pickup/dropoff delay
        self._phaseTimeout = -1
        self.isActive = True
        self.inPosition = False
        self.pickedUp = False
        self.completed = False
        self.paid = False

    def compute_fare(self) -> float:
        """
        Compute the fare earned from delivering this ducky
        :return:
        """
        return self.dist * get_distance_fare(self.type) + get_base_fare(self.type)

    def compute_karma(self) -> float:
        """
        Compute the karma earned from delivering this ducky
        :return:
        """
        return get_reputation(self.type)

    def claim_fare(self, idx: int, team: Team) -> str | None:
        """
        Claim the fare for a team
        :param idx: Index of the fare
        :param team: To claim fare for
        :return: Error message, None if successful
        """
        # Claim if not already claimed
        if self.team is not None:
            return f"Fare {idx} already claimed"
        if not self.isActive:
            return f"Fare {idx} is expired"

        self.team = team.number
        team.currentFare = idx
        return None

    def pay_fare(self, teams : list[Team]):
        """
        Pay the team their fare
        Will ensure that fare is completed and fare is not already paid
        :param teams: List of teams
        """
        if self.paid or not self.completed:
            return
        team = teams[self.team]
        if team is not None:
            team.money += self.compute_fare()
            team.karma += self.compute_karma()
            self.paid = True

    def to_json_dict(self, idx: int, extended: bool):
        data = {
            "id": idx,
            "modifiers": self.type.value,
            "src": {
                "x": self.src.x,
                "y": self.src.y
            },
            "dest": {
                "x": self.dest.x,
                "y": self.dest.y
            },
            "claimed": self.team is not None,
            "expiry": self.expiry,
            "pay": self.compute_fare(),
            "reputation": self.compute_karma()
        }
        if extended:
            data["active"] = self.isActive
            data["team"] = self.team
            data["inPosition"] = self.inPosition
            data["pickedUp"] = self.pickedUp
            data["completed"] = self.completed
            data["paid"] = self.paid
        return data

    def periodic(self, number: int, teams: list[Team]):
        """
        Update phases of the fare
        Checks team position to determine if they are at start/destination, and if dropoff/pickup should occur
        """
        # Make sure fare is paid out even if it becomes inactive
        if self.completed and not self.paid:
            self.pay_fare(teams)

        # Update active status
        if not self.isActive:
            return
        # Becomes inactive if time expires without a claiming team or the fare is completed
        self.isActive = (self.expiry > time.time() or self.team is not None) and not self.completed

        if self.team is None or self.team not in teams:
            return

        team = teams[self.team]

        # Set inactive if the team takes another fare
        if not team.currentFare == number:
            self.isActive = False

        # Check phase
        if self.pickedUp is False:
            # For pickup should be near the source position
            if team.pos.dist(self.src) < POSITION_TOLERANCE:
                self.inPosition = True
                # If no timeout started, then start it
                if self._phaseTimeout == -1:
                    self._phaseTimeout = time.time() + PICKUP_DURATION * get_load_time_multiplier(self.type)
                # If timeout completed, then set picked up
                elif self._phaseTimeout < time.time():
                    self.pickedUp = True
                    self._phaseTimeout = -1
            else:
                self.inPosition = False
                self._phaseTimeout = -1
        else:
            # For dropoff should be near the destination position
            if team.pos.dist(self.dest) < POSITION_TOLERANCE:
                self.inPosition = True
                # If no timeout started, then start it
                if self._phaseTimeout == -1:
                    self._phaseTimeout = time.time() + PICKUP_DURATION * get_load_time_multiplier(self.type)
                # If timeout completed, then set fare completed
                elif self._phaseTimeout < time.time():
                    self.completed = True
                    self._phaseTimeout = -1
            else:
                self._phaseTimeout = -1
                self.inPosition = False
