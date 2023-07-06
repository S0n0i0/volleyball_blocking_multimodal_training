from enum import Enum

class Reaction(Enum):
    NEUTRAL = 0
    HAPPY = 1
    ANGRY = 2

#Common variables
directories = { # Base directories
    "samples": "./samples/",
    "support_files": "./support_files/"
}

support_files = { # Files of the program
    "pose": { # Features file for imu_reaction_recognition
        "name": "pose.csv",
        "new": True
    },
    "face": { # Features file for face_reaction_recognition
        "name": "face.csv",
        "new": True
    },
    "pose_model": { # model file for imu_reaction_recognition
        "name": "pose.pkl",
        "new": True
    }
}

features_directories_correspondences : dict[Reaction,str] = { # Directories to take samples
    Reaction.HAPPY: directories["samples"] + "happy",
    Reaction.ANGRY: directories["samples"] + "angry",
}

window_length = 5 # Window length for imu_reaction_recognition
time_between = 2 # Time between one face and the other (neutral,happy and angry) during the face_calibration

wrists_ids = [15,16] # Wrists landmark ids [left,right]. Used for imu_reaction_recognition
eyebrows_ids = [[285,336],[55,107]] # Eyebrows landmark ids [left,right]. Used to detect anger when they go down
nose_ids = [197,195] # Nose landmark ids. Used to rescale values, since central nose measures doesn't change
corners_lips_ids = [[61,185,146],[291,375,409]] # Corner lips landmark ids [left,right]. Used to detect joy when they go up