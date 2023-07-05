# volleyball_blocking_multimodal_training

## Setup

Run this command from the base directory (`./volleybal_block_multimodal_training`):
```
pip install .
```

## IMU reaction recognition
Recognition of reactions (HAPPY,ANGRY) using machine learning classifier based on features derived from wrist accelerations during the reaction.

Unfortunately the recognition (`imu_training.py` and `imu_reaction_recognition.py`) doesn't work properly due to complications with the classifiers.
Anyway, the feature extraction works and you can set parameters in the files `data_structures.py` and `imu_features_collection.py` as you wish and run this command to do the feature extraction of the acceleration:
```
py ./src/reaction_recognition/imu_features_collection.py
```

## Face reaction recognition
Recognition of reactions using lowering eyebrows (ANGRY) and rising corner lips (HAPPY)

