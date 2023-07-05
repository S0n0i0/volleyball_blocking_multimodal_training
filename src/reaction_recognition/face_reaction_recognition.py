import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import find_peaks

from data_structures import directories,support_files,Reaction

def recognize_by_face(id, lips_positions: np.ndarray, eyebrows_positions: np.ndarray, nose_positions: np.ndarray, plots: dict[str,bool] = {}) -> Reaction:

    calibration = pd.read_csv(os.path.join(directories["support_files"],support_files["face"]["name"]))
    reference = calibration[calibration["id"] == id]
    votes = {}

    for reaction in Reaction:
        votes[reaction] = 0

    frames = np.shape(lips_positions)[0]
    base_lips = np.full((frames,2),[reference["l_l_p"][0],reference["r_l_p"][0]])
    base_eyebrows = np.full((frames,2),[reference["l_e_p"][0],reference["r_e_p"][0]])
    base_nose = np.full((frames,1),reference["n_p"][0])
    rescaled_base_lips = base_lips*nose_positions/base_nose
    rescaled_base_eyebrows = base_eyebrows*nose_positions/base_nose

    base_differences = {
        "lips": base_lips - base_nose,
        "eyebrows": base_eyebrows - base_nose
    }

    #rescaled_base_lips:nose=base_lips:base_nose
    #(lips-rescaled_base_lips):(lips-nose)=x:(base_lips-base_nose)
    differences = { #from_base:from_nose=x:from_base_nose
        "lips": (lips_positions - rescaled_base_lips)*(base_lips-base_nose)/(lips_positions-nose_positions),
        "eyebrows": (eyebrows_positions - rescaled_base_eyebrows)*(base_eyebrows-base_nose)/(eyebrows_positions-nose_positions)
    }

    '''left_peaks, _ = find_peaks(differences["lips"][:,0], height=0)
    right_peaks, _ = find_peaks(differences["lips"][:,1], height=0)'''
    
    data = False
    plt.figure(figsize=(12, 9))
    if "lips" in plots.keys() and plots["lips"]:
        plt.plot(differences["lips"][:,0], label="left_corner_lip")
        #plt.plot(left_peaks, differences["lips"][left_peaks,0], "x")
        plt.plot(differences["lips"][:,1], label="right_corner_lip")
        #plt.plot(right_peaks, differences["lips"][right_peaks,1], "x")
        #find peaks of differences["lips"] and plot them
        if "calibration" in plots.keys() and plots["calibration"]:
            plt.plot(base_differences["lips"][:,0], label="base_left_corner_lip")
            plt.plot(base_differences["lips"][:,1], label="base_right_corner_lip")
        data = True
    if "eyebrows" in plots.keys() and plots["eyebrows"]:
        plt.plot(differences["eyebrows"][:,0], label="left_eyebrow")
        plt.plot(differences["eyebrows"][:,1], label="right_eyebrow")
        if "calibration" in plots.keys() and plots["calibration"]:
            plt.plot(base_differences["eyebrows"][:,0], label="base_left_eyebrow")
            plt.plot(base_differences["eyebrows"][:,1], label="base_right_eyebrow")
        data = True
    plt.grid(True)
    if data:
        plt.legend()
        plt.show()