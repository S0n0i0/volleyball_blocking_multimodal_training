import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from data_structures import directories,support_files,eyebrows_ids,corners_lips_ids,nose_ids,Reaction,time_between
from utils import export_to_csv,get_features

def face_calibration(output_files, id, time_between, plots: dict[str,bool|Reaction] = {}):

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions

    # Initialize calibration file
    if support_files["face"]["new"]:
        title_row = ["id","n_n","l_l_n","r_l_n","l_e_n","r_e_n","l_l_h","r_l_h","l_e_l","r_e_l"]
        export_to_csv(os.path.join(directories["support_files"],output_files["face"]["name"]),"w",title_row)
        support_files["face"]["new"] = False

    calibration_row = []
    lips_positions = {}
    eyebrows_positions = {}
    nose_positions = {}
    for reaction in Reaction:
        count = 0
        print(f"Do a {reaction.name} face")
        time.sleep(time_between)

        cap = cv2.VideoCapture(0)
        lips_positions[reaction] = []
        eyebrows_positions[reaction] = []
        nose_positions[reaction] = []
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if ret:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False        
                    
                    # Make Detections
                    results = holistic.process(image)

                    # Recolor image back to BGR for rendering
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Face Detections
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )

                    try:
                        # Extract Face landmarks
                        face = results.face_landmarks.landmark
                        #center_positions += [[face[landmark_id].y for landmark_id in center_lips]]
                        lips_positions[reaction] += [get_features(face,corners_lips_ids,"y",np.mean)]
                        eyebrows_positions[reaction] += [get_features(face,eyebrows_ids,"y",np.mean)]
                        nose_positions[reaction] += [get_features(face,nose_ids,"y",np.mean)]

                    except Exception as err:
                        print(err)
                        pass
                            
                    cv2.imshow('Monitoring video', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                    
                    # Frame cap
                    if count < 10:
                        count += 1
                    else:
                        break
        
        cap.release()
        cv2.destroyAllWindows()

        lips_positions[reaction] = np.array(lips_positions[reaction])
        eyebrows_positions[reaction] = np.array(eyebrows_positions[reaction])
        nose_positions[reaction] = np.array(nose_positions[reaction])
        
        # Add calibration
        if reaction == Reaction.NEUTRAL:
            calibration_row += [id,
                                np.mean(nose_positions[reaction][:,0]),
                                np.mean(lips_positions[reaction][:,0]),np.mean(lips_positions[reaction][:,1]),
                                np.mean(eyebrows_positions[reaction][:,0]),np.mean(eyebrows_positions[reaction][:,1])]
        elif reaction == Reaction.HAPPY:
            calibration_row += [np.mean(lips_positions[reaction][:,0]),np.mean(lips_positions[reaction][:,1])]
        elif reaction == Reaction.ANGRY:
            calibration_row += [np.mean(eyebrows_positions[reaction][:,0]),np.mean(eyebrows_positions[reaction][:,1])]
    
    export_to_csv(os.path.join(directories["support_files"],output_files["face"]["name"]),"a",calibration_row)

    # Plot calibration
    data = False
    plt.figure(figsize=(12, 9))
    #plt.plot(center_positions[:,0], label="center")
    reaction_plot = Reaction.NEUTRAL if "reaction" not in plots.keys() else plots["reaction"]
    if "nose" in plots.keys() and plots["nose"]:
        plt.plot(nose_positions[reaction_plot][:,0], label="nose")
        data = True
    if "lips" in plots.keys() and plots["lips"]:
        plt.plot(lips_positions[reaction_plot][:,0], label="left_corner_lip")
        plt.plot(lips_positions[reaction_plot][:,1], label="right_corner_lip")
        data = True
    if "eyebrows" in plots.keys() and plots["eyebrows"]:
        plt.plot(eyebrows_positions[reaction_plot][:,0], label="left_eyebrow")
        plt.plot(eyebrows_positions[reaction_plot][:,1], label="right_eyebrow")
        data = True
    plt.grid(True)
    if data:
        plt.legend()
        plt.show()

# Data to visualize in the plot
plots = {
    "nose": True,
    "lips": True,
    "eyebrows": True,
    "reaction": Reaction.NEUTRAL
}

print("How many people do you want to calibrate? ")
n = int(input())
for i in range(n):
    print("Who are you calibrating? ")
    id = input()
    face_calibration(support_files,id,time_between,plots)