from typing import NamedTuple
from enum import Enum

# Common classes
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

#Common variables
directories = {
    "samples": "./samples/",
    "support_files": "./support_files/"
}

support_files = {
    "pose": {
        "name": "pose.csv",
        "new": True
    },
    "face": {
        "name": "face.csv",
        "new": True
    },
}

features_directories_correspondences : dict[Reaction,str] = {
    Reaction.HAPPY: directories["samples"] + "happy",
    Reaction.ANGRY: directories["samples"] + "angry",
}

window_length = 5

wrists_ids = [15,16] # Wrists landmark ids [left,right]
eyebrows_ids = [[285,336,295,296,282,334],[55,107,65,66,52,105]] # Eyebrows landmark ids [left,right]
nose_ids = [197,195]
corners_lips_ids = [[61,185,146],[291,375,409]] # Corner lips landmark ids [left,right]