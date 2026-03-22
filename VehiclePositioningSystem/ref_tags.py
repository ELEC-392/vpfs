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
# Positions are in CENTIMETRES (x, y), measured from any convenient physical origin.
#
# Tag 95 marks one corner of the field; tag 96 lies along the +X direction
# from tag 95 and defines the orientation of the X axis.  Tag 95 does NOT
# need to be placed at (0, 0) — set all coordinates to the actual physical
# measurements from your chosen origin.
#
# The more reference tags a camera sees, the more stable its pose estimate.

_addTag(ReferenceTag(85, 551.0, 255.5))
_addTag(ReferenceTag(86, 152.0, 92.5))
_addTag(ReferenceTag(87, 76.0, 260.5))
_addTag(ReferenceTag(88, 400.0, 90.5))
_addTag(ReferenceTag(89, 549.0, 157.5)) 
_addTag(ReferenceTag(90, 401.0, 257.5)) 
_addTag(ReferenceTag(91, 217.0, 408.5)) 
_addTag(ReferenceTag(92, 481.0, 343.5)) 
_addTag(ReferenceTag(93, 280.0, 425.5)) 
_addTag(ReferenceTag(94, 315.0, 344.5))  
_addTag(ReferenceTag(95, 73.0, 73.0))        
_addTag(ReferenceTag(96, 515.0, 73.0))      
_addTag(ReferenceTag(97, 516.0, 430.0))    
_addTag(ReferenceTag(98, 102.0, 402.5))      
_addTag(ReferenceTag(99, 254.0, 263.5))        