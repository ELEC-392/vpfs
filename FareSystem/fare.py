import random
import sys
from pathlib import Path

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

# Recorder is optional — fare.py stays functional even without it.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Databases"))
    import recorder as _recorder
except ImportError:
    _recorder = None


def _rec_event(
    fare_uid: int,
    event_type: str,
    team_id=None,
    team_x=None,
    team_y=None,
    money_after=None,
    karma_after=None,
) -> None:
    """Fire-and-forget recorder call; silently skipped when recorder is absent."""
    if _recorder is not None:
        _recorder.record_event(
            fare_uid, fare_uid // 1000, team_id, event_type,
            team_x, team_y, money_after, karma_after,
        )

class Fare:
    def __init__(self, src : Point, dest: Point, fare_type: FareType, match_num: int = 0, sequence: int = 0):
        """
        :param src: Location the ducky is picked up at
        :param dest: Location the ducky is delivered to
        :param fare_type: Type of fare (STANDARD, SPECIAL, etc.)
        :param match_num: Match number for unique ID generation
        :param sequence: Sequence number within the match for unique ID
        """
        self.src = src
        self.dest = dest
        self.dist = src.dist(dest)
        self.type = fare_type
        
        # Generate unique ID: match_num * 1000 + sequence
        # This allows up to 999 fares per match with globally unique IDs
        self.unique_id = match_num * 1000 + sequence
        
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
        if team.currentFare is not None:
            return f"Team {team.number} already has an active fare"

        self.team = team.number
        team.currentFare = idx
        _rec_event(self.unique_id, "CLAIMED", team_id=team.number,
                   team_x=team.pos.x, team_y=team.pos.y)
        return None

    def drop_fare(self, idx: int, team: Team) -> str | None:
        """
        Drop a previously claimed fare, returning it to the pool.
        Teams may drop a fare at any time. If the fare has already been picked up
        (i.e. it is in progress), the fare's full reputation value is deducted from
        the team's karma as a penalty. No penalty is applied if the fare has not
        yet been picked up.
        :param idx: Index of the fare
        :param team: Team attempting to drop the fare
        :return: Error message, None if successful
        """
        if self.team != team.number:
            return f"Fare {idx} is not claimed by team {team.number}"

        if self.pickedUp:
            team.karma -= self.compute_karma()
            team.karma = max(-100, min(100, team.karma))

        self.team = None
        self.inPosition = False
        self.pickedUp = False
        self._phaseTimeout = -1
        team.currentFare = None
        _rec_event(self.unique_id, "DROPPED", team_id=team.number,
                   team_x=team.pos.x, team_y=team.pos.y,
                   money_after=team.money, karma_after=team.karma)
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
            team.karma = max(-100, min(100, team.karma))
            team.currentFare = None
            self.paid = True
            _rec_event(self.unique_id, "PAID", team_id=self.team,
                       money_after=team.money, karma_after=team.karma)

    def to_json_dict(self, idx: int, extended: bool):
        data = {
            "id": idx,  # Keep for backwards compatibility with list index
            "unique_id": self.unique_id,  # Globally unique ID for database
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
        # Becomes inactive if time expires without a claiming team or the fare is completed.
        # Note: a fare held by a team (self.team is not None) stays active until dropped/completed.
        self.isActive = (self.expiry > time.time() or self.team is not None) and not self.completed
        if not self.isActive:
            # Only reachable when self.team is None and the expiry has passed.
            _rec_event(self.unique_id, "EXPIRED")

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
                    # First tick in pickup zone — record arrival.
                    _rec_event(self.unique_id, "AT_PICKUP_ZONE", team_id=self.team,
                               team_x=team.pos.x, team_y=team.pos.y)
                # If timeout completed, then set picked up
                elif self._phaseTimeout < time.time():
                    self.pickedUp = True
                    self._phaseTimeout = -1
                    _rec_event(self.unique_id, "LOADED", team_id=self.team,
                               team_x=team.pos.x, team_y=team.pos.y)
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
                    # First tick in dropoff zone — record arrival.
                    _rec_event(self.unique_id, "AT_DROPOFF_ZONE", team_id=self.team,
                               team_x=team.pos.x, team_y=team.pos.y)
                # If timeout completed, then set fare completed
                elif self._phaseTimeout < time.time():
                    self.completed = True
                    self._phaseTimeout = -1
                    _rec_event(self.unique_id, "DELIVERED", team_id=self.team,
                               team_x=team.pos.x, team_y=team.pos.y)
            else:
                self._phaseTimeout = -1
                self.inPosition = False
