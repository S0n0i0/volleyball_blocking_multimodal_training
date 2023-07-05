import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

from data_structures import directories,window_length,wrists_ids,eyebrows_ids,corners_lips_ids,nose_ids
from utils import get_features
from imu_reaction_recognition import recognize_by_imu
from face_reaction_recognition import recognize_by_face

def detect_reaction(source,window_length,plots: dict[str,bool] = {}):

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions

    cap = cv2.VideoCapture(source)
    pose_positions = []
    pose_visibility = []
    lips_positions = []
    eyebrows_positions = []
    nose_distances = []

    count = 0
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
                
                # Face Detections
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )

                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_positions += [[[landmark.x, landmark.y, landmark.z] for landmark in pose]]
                    pose_visibility += [[[landmark.visibility] for landmark in pose]]

                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
                    lips_positions += [get_features(face,corners_lips_ids,"y",np.mean)]
                    eyebrows_positions += [get_features(face,eyebrows_ids,"y",np.mean)]
                    nose_distances += [[face[nose_ids[0]].y-face[nose_ids[1]].y]]
                    
                    #face_row = list(np.array([[face[landmark].y, face[landmark].visibility] for landmark in np.array(eyebrows_id).flatten()]).flatten())
                    
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
    
    cap.release()
    cv2.destroyAllWindows()

    lips_positions = np.array(lips_positions)
    eyebrows_positions = np.array(eyebrows_positions)
    nose_distances = np.array(nose_distances)
    recognize_by_face(0,lips_positions,eyebrows_positions,nose_distances,plots)

    pose_positions = np.array(pose_positions)
    pose_visibility = np.array(pose_visibility)
    pose_reaction_class, pose_reaction_prob = recognize_by_imu(window_length,wrists_ids,pose_positions,pose_visibility)

plots = {
    "lips": True,
    "eyebrows": True,
    "calibration": True,
}

detect_reaction(0,window_length,plots)