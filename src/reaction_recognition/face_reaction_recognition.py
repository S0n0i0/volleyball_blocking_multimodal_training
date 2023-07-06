import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import find_peaks

from data_structures import directories,support_files,Reaction

def recognize_by_face(id, lips_positions: np.ndarray, eyebrows_positions: np.ndarray, nose_positions: np.ndarray, plots: dict[str,bool] = {}) -> Reaction: # Given position data return the reaction

    # Import calibration
    calibration = pd.read_csv(os.path.join(directories["support_files"],support_files["face"]["name"]))
    reference = calibration[calibration["id"] == id]
    votes = {}

    # Initializa votes
    for reaction in Reaction:
        votes[reaction] = 0

    # Prepare calibration values
    frames = np.shape(lips_positions)[0]
    neutral_lips = np.full((frames,2),[reference["l_l_n"][0],reference["r_l_n"][0]])
    neutral_eyebrows = np.full((frames,2),[reference["l_e_n"][0],reference["r_e_n"][0]])
    neutral_nose = np.full((frames,1),reference["n_n"][0])
    high_lips = np.full((frames,2),[reference["l_l_h"][0],reference["r_l_h"][0]])
    low_eyebrows = np.full((frames,2),[reference["l_e_l"][0],reference["r_e_l"][0]])
    
    #rescaled_base_lips:nose = base_lips:base_nose
    rescaled_base_lips = neutral_lips*nose_positions/neutral_nose
    #rescaled_base_eyebrows:nose = base_eyebrows:base_nose
    rescaled_base_eyebrows = neutral_eyebrows*nose_positions/neutral_nose
    base_nose_differences = {
        "neutral_lips": neutral_lips - neutral_nose,
        "neutral_eyebrows": neutral_eyebrows - neutral_nose,
        "high_lips": high_lips - neutral_nose,
        "low_eyebrows": low_eyebrows - neutral_nose
    }

    # Compute differences using rescaled values
    #(lips-rescaled_base_lips):(lips-nose)=x:(base_lips-base_nose)
    rescaled_differences = {
        "lips": (lips_positions - rescaled_base_lips)*(base_nose_differences["neutral_lips"])/(lips_positions-nose_positions),
        "eyebrows": (eyebrows_positions - rescaled_base_eyebrows)*(base_nose_differences["neutral_eyebrows"])/(eyebrows_positions-nose_positions)
    }

    '''# Hypotesis: use find_peaks
    left_peaks, _ = find_peaks(differences["lips"][:,0], height=0)
    right_peaks, _ = find_peaks(differences["lips"][:,1], height=0)'''
    
    # Plot data
    data = False
    plt.figure(figsize=(12, 9))
    if "lips" in plots.keys() and plots["lips"]:
        plt.plot(rescaled_differences["lips"][:,0], label="rescaled_left_corner_lip")
        #plt.plot(left_peaks, differences["lips"][left_peaks,0], "x")
        plt.plot(rescaled_differences["lips"][:,1], label="rescaled_right_corner_lip")
        #plt.plot(right_peaks, differences["lips"][right_peaks,1], "x")
        #find peaks of differences["lips"] and plot them
        if "calibration" in plots.keys() and plots["calibration"]:
            plt.plot(base_nose_differences["neutral_lips"][:,0], label="neutral_nose-left_corner_lip")
            plt.plot(base_nose_differences["neutral_lips"][:,1], label="neutral_nose-right_corner_lip")
            #plt.plot(base_nose_differences["high_lips"][:,0], label="high_nose-left_corner_lip")
            #plt.plot(base_nose_differences["high_lips"][:,1], label="high_nose-right_corner_lip")
        data = True
    if "eyebrows" in plots.keys() and plots["eyebrows"]:
        plt.plot(rescaled_differences["eyebrows"][:,0], label="rescaled_left_eyebrow")
        plt.plot(rescaled_differences["eyebrows"][:,1], label="rescaled_right_eyebrow")
        if "calibration" in plots.keys() and plots["calibration"]:
            plt.plot(base_nose_differences["neutral_eyebrows"][:,0], label="neutral_nose-left_eyebrow")
            plt.plot(base_nose_differences["neutral_eyebrows"][:,1], label="neutral_nose-right_eyebrow")
            #plt.plot(base_nose_differences["low_eyebrows"][:,0], label="low_nose-left_eyebrow")
            #plt.plot(base_nose_differences["low_eyebrows"][:,1], label="low_nose-right_eyebrow")
        data = True
    plt.grid(True)
    if data:
        plt.legend()
        plt.show()