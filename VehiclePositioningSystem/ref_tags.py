from typing import Dict
import numpy as np
from numpy.typing import *

class ReferenceTag:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.z = 0
        self.mat = self._build_mat()

    def _build_mat(self) -> ArrayLike:
        return np.array([
            [1, 0, 0, self.x],
            [0, -1, 0, self.y],
            [0, 0, -1, self.z],
            [0, 0, 0, 1]
        ])

ref_tags: Dict[int, ReferenceTag] = { }

def _addTag(tag: ReferenceTag):
    ref_tags[tag.id] = tag


# Tags for lab setup
# _addTag(ReferenceTag(584, 0.53, 0.715))

# For the map we are using IDs: 95, 96, 97, 98, 99
_addTag(ReferenceTag(95, 0., 0.))
_addTag(ReferenceTag(96, 0.6, 0.))
_addTag(ReferenceTag(97, 0.6, 0.7))
_addTag(ReferenceTag(98, 0., 0.7))
_addTag(ReferenceTag(99, 0.4, 0.36))