from typing import NamedTuple
from enum import Enum

#Common variables
directory = "./samples/"
features_files = {
    "pose": {
        "name": "pose.csv",
        "new": True
    },
    "face": {
        "name": "face.csv",
        "new": True
    },
}

wrists_id = [15,16] # Wrists landmark ids [left,right]

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