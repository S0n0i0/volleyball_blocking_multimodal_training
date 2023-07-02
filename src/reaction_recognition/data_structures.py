from typing import NamedTuple
from enum import Enum

class Coordinate(NamedTuple):
    x: float
    y: float
    z: float

class Acceleration3D(NamedTuple):
    x: float
    y: float
    z: float

class Reaction(Enum):
    HAPPY = 1
    ANGRY = 2
    TENSE = 3