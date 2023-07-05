import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

from data_structures import directories,window_length,wrists_ids,eyebrows_ids,corners_lips_ids,nose_ids
from utils import get_acceleration_features

def detect_reaction(source,window_length,model):

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions

    cap = cv2.VideoCapture(source)
    pose_positions = []
    pose_visibility = []
    pose_row = []
    lips_positions = []
    lips_visibility = []
    face_row = []
    visibility_row = []
    center_positions = []
    nose_positions = []
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
                    #center_positions += [[face[landmark_id].y for landmark_id in center_lips]] 
                    lips_positions += [[#np.mean([face[landmark_id].y for landmark_id in center_lips]),
                                 np.mean([face[landmark_id].y for landmark_id in corners_lips_ids[0]]),
                                 np.mean([face[landmark_id].y for landmark_id in corners_lips_ids[1]])]]
                    
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

    center_positions = np.array(center_positions)
    lips_positions = np.array(lips_positions)
    print(np.shape(center_positions))
    plt.figure(figsize=(12, 9))
    '''plt.plot(lips_positions[:,0], label="center")
    plt.plot(center_positions[:,0], label="center_up")
    plt.plot(center_positions[:,1], label="center_down")
    plt.plot(nose_positions[:], label="nose")'''
    plt.plot(lips_positions[:,1], label="left")
    plt.plot(lips_positions[:,2], label="right")
    plt.legend()
    plt.grid(True)
    plt.show()

    pose_positions = np.array(pose_positions)
    pose_visibility = np.array(pose_visibility)
    pose_X = pd.DataFrame(get_acceleration_features(window_length,wrists_ids,pose_positions,pose_visibility))
    pose_reaction_class = model.predict(pose_X)[0]
    pose_reaction_prob = model.predict_proba(pose_X)[0]
    print(pose_reaction_class, pose_reaction_prob)

with open(os.path.join(directories["support_files"],'pose.pkl'), 'rb') as f:
    model = pickle.load(f)

detect_reaction(0,window_length,model)