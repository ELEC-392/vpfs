import random
from typing import Self

from fare_types import FareType


class FareProbability:
    def __init__(self, values: dict[FareType, float] | None = None):
        # Default to a uniform distribution across configured fare types
        self._values: dict[FareType, float] = {fare_type: 1.0 for fare_type in FareType}
        if values:
            for fare_type, weight in values.items():
                self._values[fare_type] = float(weight)

    @staticmethod
    def merge(p1, p2):
        return FareProbability({
            fare_type: p1._values[fare_type] + p2._values[fare_type]
            for fare_type in FareType
        })

    def copy(self) -> Self:
        return FareProbability({fare_type: weight for fare_type, weight in self._values.items()})

    def __getitem__(self, key: FareType):
        return self._values[key]

    def __setitem__(self, key: FareType, value: float):
        self._values[key] = value

    def keys(self):
        return self._values.keys()

    def values(self):
        return self._values.values()

    def __iter__(self):
        return self._values.items().__iter__()

    def roll(self) -> FareType:
        """
        Roll the probability, and obtain resulting fare type
        :return: Faretype selected based on probabilities
        """
        types: [FareType] = []
        probs: [float] = []
        total = 0

        # Create collections of keys/weights, also handle negatives here
        for key, value in self._values.items():
            types.append(key)
            prob = max(0.0, value)
            probs.append(prob)
            total += prob

        # If the probabilities sum to zero, then don't roll
        if total == 0:
            return next(iter(FareType))

        # Use weighted random function
        return random.choices(types, probs)[0]



