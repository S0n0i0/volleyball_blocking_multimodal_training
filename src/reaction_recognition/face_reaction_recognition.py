import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from data_structures import directories,support_files,Reaction

def recognize_by_face(id, lips_positions: np.ndarray, eyebrows_positions: np.ndarray, nose_distances: np.ndarray, plots: dict[str,bool] = {}) -> Reaction:

    calibration = pd.read_csv(os.path.join(directories["support_files"],support_files["face"]["name"]))
    reference = calibration[calibration["id"] == id]
    votes = {}
    differences = {
        "lips": [[],[]],
        "eyebrows": [[],[]]
    }

    for reaction in Reaction:
        votes[reaction] = 0

    frames = np.shape(lips_positions)[0]
    base_lips = np.full((frames,2),[reference["l_l_p"][0],reference["r_l_p"][0]])
    base_eyebrows = np.full((frames,2),[reference["l_e_p"][0],reference["r_e_p"][0]])
    differences = {
        "lips": lips_positions - base_lips,
        "eyebrows": eyebrows_positions - base_eyebrows
    }
    '''for position in range(frames):
        differences["lips"][0] += [lips_positions[position][0] - reference["l_l_p"]]
        differences["lips"][1] += [lips_positions[position][1] - reference["r_l_p"]]
        differences["eyebrows"][0] += [eyebrows_positions[position][0] - reference["l_e_p"]]
        differences["eyebrows"][1] += [eyebrows_positions[position][1] - reference["r_e_p"]]'''
    
    data = False
    plt.figure(figsize=(12, 9))
    if "lips" in plots.keys() and plots["lips"]:
        plt.plot(lips_positions[:,0], label="left_corner_lip")
        plt.plot(lips_positions[:,1], label="right_corner_lip")
        if "calibration" in plots.keys() and plots["calibration"]:
            plt.plot(base_lips[:,0], label="base_left_corner_lip")
            plt.plot(base_lips[:,1], label="base_right_corner_lip")
        data = True
    if "eyebrows" in plots.keys() and plots["eyebrows"]:
        plt.plot(eyebrows_positions[:,0], label="left_eyebrow")
        plt.plot(eyebrows_positions[:,1], label="right_eyebrow")
        if "calibration" in plots.keys() and plots["calibration"]:
            plt.plot(base_eyebrows[:,0], label="base_left_eyebrow")
            plt.plot(base_eyebrows[:,1], label="base_right_eyebrow")
        data = True
    plt.grid(True)
    if data:
        plt.legend()
        plt.show()