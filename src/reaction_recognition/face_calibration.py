import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import os
import matplotlib.pyplot as plt

from data_structures import directories,support_files,eyebrows_ids,corners_lips_ids,nose_ids
from utils import export_to_csv,get_features

def face_calibration(output_files, id, plots: dict[str,bool] = {}):

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions

    cap = cv2.VideoCapture(0)
    lips_positions = []
    eyebrows_positions = []
    nose_positions = []

    count = 0
    print("Do a neautral face")
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
                    lips_positions += [get_features(face,corners_lips_ids,"y",np.mean)]
                    eyebrows_positions += [get_features(face,eyebrows_ids,"y",np.mean)]
                    nose_positions += [get_features(face,nose_ids,"y",np.mean)]

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

    lips_positions = np.array(lips_positions)
    eyebrows_positions = np.array(eyebrows_positions)
    nose_positions = np.array(nose_positions)
    
    # Initialize calibration file
    if support_files["face"]["new"]:
        title_row = ["id","n_p","l_l_p","r_l_p","l_e_p","r_e_p"]
        export_to_csv(os.path.join(directories["support_files"],output_files["face"]["name"]),"w",title_row)
        support_files["face"]["new"] = False
    
    # Add calibration
    calibration_row = [id,
                       #np.mean([face[landmark_id].y for landmark_id in center_lips]),
                       np.mean(nose_positions[:,0]),
                       np.mean(lips_positions[:,0]),np.mean(lips_positions[:,1]),
                       np.mean(eyebrows_positions[:,0]),np.mean(eyebrows_positions[:,1])]
    export_to_csv(os.path.join(directories["support_files"],output_files["face"]["name"]),"a",calibration_row)

    # Plot calibration
    data = False
    plt.figure(figsize=(12, 9))
    #plt.plot(center_positions[:,0], label="center")
    if "nose" in plots.keys() and plots["nose"]:
        plt.plot(nose_positions[:,0], label="nose")
        data = True
    if "lips" in plots.keys() and plots["lips"]:
        plt.plot(lips_positions[:,0], label="left_corner_lip")
        plt.plot(lips_positions[:,1], label="right_corner_lip")
        data = True
    if "eyebrows" in plots.keys() and plots["eyebrows"]:
        plt.plot(eyebrows_positions[:,0], label="left_eyebrow")
        plt.plot(eyebrows_positions[:,1], label="right_eyebrow")
        data = True
    plt.grid(True)
    if data:
        plt.legend()
        plt.show()

# Data to visualize in the plot
plots = {
    "nose": True,
    "lips": True,
    "eyebrows": True
}

print("How many people do you want to calibrate? ")
n = int(input())
for i in range(n):
    print("Who are you calibrating? ")
    id = input()
    face_calibration(support_files,id,plots)