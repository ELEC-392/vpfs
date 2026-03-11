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
        # This is the map_to_tag transform: p_tag = R @ p_map + t
        # For tag origin (p_tag=[0,0,0]) to sit at world position (x, y, z):
        #   t = -R @ [x, y, z]
        #
        # Physical setup: markers lie flat on the floor, face pointing UP toward
        # the ceiling camera. World frame: X=right, Y=forward, Z=up (ceiling).
        # The tag axes align with the world axes for all uniformly-placed markers:
        #   tag_X = world_X  =>  row (1, 0, 0)
        #   tag_Y = world_Y  =>  row (0, 1, 0)
        #   tag_Z = world_Z  =>  row (0, 0, 1)
        # So R = Identity, and t = -I @ [x, y, z] = [-x, -y, -z].
        return np.array([
            [1, 0, 0, -self.x],
            [0, 1, 0, -self.y],
            [0, 0, 1, -self.z],
            [0, 0, 0,  1     ]
        ])

ref_tags: Dict[int, ReferenceTag] = { }

def _addTag(tag: ReferenceTag):
    ref_tags[tag.id] = tag


# Tags for lab setup
# _addTag(ReferenceTag(584, 0.53, 0.715))

# Reference tags (95-99) - Fixed markers with known world positions.
# Positions are in metres (x, y), measured from any convenient physical origin.
#
# Tag 95 marks one corner of the field; tag 96 lies along the +X direction
# from tag 95 and defines the orientation of the X axis.  Tag 95 does NOT
# need to be placed at (0, 0) — set all coordinates to the actual physical
# measurements from your chosen origin.
#
# The more reference tags a camera sees, the more stable its pose estimate.

_addTag(ReferenceTag(95, 0., 0.))      # corner marker
_addTag(ReferenceTag(96, 1.05, 0.))    # +X axis reference (1.05 m from tag 95)
_addTag(ReferenceTag(97, 1.05, 1.05))  # far-right corner
_addTag(ReferenceTag(98, 0., 1.05))    # far-left corner
_addTag(ReferenceTag(99, 0.5, 0.))     # centre-line marker