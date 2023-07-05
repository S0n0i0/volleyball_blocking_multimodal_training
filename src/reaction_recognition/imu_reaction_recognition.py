import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd

from data_structures import directories,support_files,Reaction
from utils import compute_acceleration,export_to_csv

def get_acceleration_features(window_length: int, wrists_id: list[int], pose_positions: np.ndarray, pose_visibility: np.ndarray, plots: dict[str,bool] = {}, label = None, directories: dict = {}, output_files: dict = {}) -> list:

    export = False
    if len(directories.keys()) > 0 and len(output_files.keys()) > 0:
        export = True
    
    accelerations = compute_acceleration(pose_positions)

    # Plot the positions over time
    if "positions" in plots.keys() and plots["positions"]:
        plt.figure(figsize=(12, 9))
        plt.plot(pose_positions[:,15,0], label="left_x")
        plt.plot(pose_positions[:,15,1], label="left_y")
        plt.plot(pose_positions[:,15,2], label="left_z")
        plt.plot(pose_positions[:,16,0], label="right_x")
        plt.plot(pose_positions[:,16,1], label="right_y")
        plt.plot(pose_positions[:,16,2], label="right_z")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Plot the acceleration over time
    if "accelerations" in plots.keys() and plots["accelerations"]:
        plt.figure(figsize=(12, 9))
        plt.plot(np.multiply(accelerations[:,0,15],pose_visibility[:,15,0]), label="left")
        plt.plot(np.multiply(accelerations[:,0,16],pose_visibility[:,16,0]), label="right")
        plt.legend()
        plt.grid(True)
        plt.show()

    common_columns = {}
    for landmark in wrists_id:
        common_columns[landmark] = [np.mean(accelerations[:,0,landmark]), # Mean of all accelerations
                                    np.average(accelerations[:,0,landmark],weights=pose_visibility[:,landmark,0])] # Mean of all accelerations, weighted with visibility

    pose_X = []
    for point in range(np.shape(pose_positions)[0]):
        pose_row = [] if label is None else [label]
        for landmark in wrists_id:
            start = (point-window_length+1)
            pose_row += [accelerations[point,0,landmark], # Single point acceleration
                         np.mean(accelerations[start if start >= 0 else 0:point+1,0,landmark]), # Mean of <window_length> accelerations
                         np.average(accelerations[start if start >= 0 else 0:point+1,0,landmark],weights=pose_visibility[start if start >= 0 else 0:point+1,landmark,0])] # Mean of <window_length> accelerations, weighted with visibility
            pose_row += common_columns[landmark]
        pose_X += [pose_row]
        if export:
            export_to_csv(os.path.join(directories["support_files"],output_files["pose"]["name"]),"a",pose_row)
    
    return pose_X

def recognize_by_imu(window_length: int, wrists_ids: list, pose_positions: np.ndarray, pose_visibility: np.ndarray) -> list[Reaction,float]:
    
    with open(os.path.join(directories["support_files"],support_files["pose_model"]["name"]), "rb") as f:
        model = pickle.load(f)
        
        pose_X = pd.DataFrame(get_acceleration_features(window_length,wrists_ids,pose_positions,pose_visibility))
        reaction_class = model.predict(pose_X)[0]
        reaction_prob = model.predict_proba(pose_X)[0]

        return [reaction_class,reaction_prob]