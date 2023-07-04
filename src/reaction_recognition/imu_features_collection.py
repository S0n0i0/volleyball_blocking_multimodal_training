import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import os

from utils import export_to_csv
from data_structures import Reaction,directories,wrists_id,features_files,window_length,features_directories_correspondences
from imu_reaction_recognition import get_acceleration_features

def collect_features(source: str | int, output_files, window_length: int, label: Reaction):

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions

    # Pose header
    if output_files["pose"]["new"]:
        landmarks = ["class"]
        for wrist in ["l","r"]:
            landmarks += [f"{wrist}_a",f"{wrist}_wm_a",f"{wrist}_wm_av",f"{wrist}_m_a",f"{wrist}_m_av"]
        export_to_csv(os.path.join(directories["support_files"],output_files["pose"]["name"]),"w",landmarks)
        output_files["pose"]["new"] = False

    #count = 0
    cap = cv2.VideoCapture(source)
    positions = []
    visibility = []
    pose_row = []
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
            
                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )

                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    positions += [[[landmark.x, landmark.y, landmark.z] for landmark in pose]]
                    visibility += [[[landmark.visibility] for landmark in pose]]
                    
                except Exception as err:
                    print(err)
                    pass
                cv2.imshow('Monitoring video', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                # Frame cap
                '''if count < 10:
                    count += 1
                else:
                    break'''
            else:
                break
    
    cap.release()
    cv2.destroyAllWindows()

    # Hypotesis: (frame,point,dims) -> (point,dims,frame)
    positions = np.array(positions)
    visibility = np.array(visibility)
    get_acceleration_features(window_length,wrists_id,positions,visibility,plots,label,directories,output_files)

fromWebCam = False
label = "hi"
plots = {
    "positions":False,
    "accelerations":False
}

if fromWebCam:
    collect_features(0,features_files,window_length,label)
else:
    for reaction_label in features_directories_correspondences.keys():
        # List of file in a directory
        input_files = [f for f in os.listdir(features_directories_correspondences[reaction_label]) if os.path.isfile(os.path.join(features_directories_correspondences[reaction_label], f))]

        for file in input_files:
            path = os.path.join(features_directories_correspondences[reaction_label], file)
            print(f"Analyzing: {path} ({reaction_label.name})")
            collect_features(path,features_files,window_length,reaction_label.name)
            print("Done")