# volleyball_blocking_multimodal_training

## Setup

Run this command from the base directory (`./volleybal_block_multimodal_training`):
```
pip install .
```

Files `data_structures.py` and `utils.py` contains common elements for the project.

## IMU reaction recognition
Recognition of reactions (HAPPY,ANGRY) using machine learning classifier based on features derived from wrist accelerations during the reaction.

Unfortunately the recognition (`imu_training.py` and `imu_reaction_recognition.py`) doesn't work properly due to complications with the classifiers. The recognition is implemented in `reaction_recognition.py`, but commented.
Anyway, the feature extraction works and you can set parameters in the files `data_structures.py` and `imu_features_collection.py` as you wish and run this command to do the feature extraction of the acceleration:
```
py ./src/reaction_recognition/imu_features_collection.py
```

## Face reaction recognition
Recognition of reactions using lowering eyebrows (ANGRY) and rising corner lips (HAPPY).

The implemented part consists in:

* Calibration (parameters in `data_structures.py` and `face_calibration.py`): calibrate as many people as you want running this command and following the instructions
```
py ./src/reaction_recognition/face_calibration.py
```
* Features collection (parameters in `data_structures.py` and `face_reaction_recognition.py`): run this command to see the features collected for the face_reaction_recognition
```
py ./src/reaction_recognition/reaction_recognition.py
```

To make the face_reaction_recognition work should be developed a dynamic threshold solution, since the distances are little and it's difficult to solve this problem with a fixed threshold.