import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

from data_structures import directories,wrists_id,eyebrows_id,corners_lips_id,window_length
from utils import compute_acceleration,export_to_csv

def detect_reaction(source,window_length,model):

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions

    cap = cv2.VideoCapture(source)
    pose_positions = []
    pose_visibility = []
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
                
                # Face Detections
                '''mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )'''

                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_positions += [[[landmark.x, landmark.y, landmark.z] for landmark in pose]]
                    pose_visibility += [[[landmark.visibility] for landmark in pose]]

                    # Extract Face landmarks
                    '''face = results.face_landmarks.landmark
                    face_row = list(np.array([[face[landmark].y, face[landmark].visibility] for landmark in np.array(eyebrows_id).flatten()]).flatten())'''
                    
                except NameError:
                    print(NameError)
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

    pose_positions = np.array(pose_positions)
    pose_visibility = np.array(pose_visibility)
    
    # Plot the positions over time
    '''plt.figure(figsize=(12, 9))
    plt.plot(pose_positions[:,15,0], label="left_x")
    plt.plot(pose_positions[:,15,1], label="left_y")
    plt.plot(pose_positions[:,15,2], label="left_z")
    plt.plot(pose_positions[:,16,0], label="right_x")
    plt.plot(pose_positions[:,16,1], label="right_y")
    plt.plot(pose_positions[:,16,2], label="right_z")
    plt.legend()
    plt.grid(True)
    plt.show()'''

    accelerations = compute_acceleration(pose_positions)
    
    # Plot the acceleration over time
    '''plt.figure(figsize=(12, 9))
    plt.plot(np.multiply(accelerations[:,0,15],pose_visibility[:,15,0]), label="left")
    plt.plot(np.multiply(accelerations[:,0,16],pose_visibility[:,16,0]), label="right")
    plt.legend()
    plt.grid(True)
    plt.show()'''

    common_columns = {}
    for landmark in wrists_id:
        common_columns[landmark] = [np.mean(accelerations[:,0,landmark]), # Mean of all accelerations
                                    np.average(accelerations[:,0,landmark],weights=pose_visibility[:,landmark,0])] # Mean of all accelerations, weighted with visibility

    pose_X = []
    for point in range(np.shape(pose_positions)[0]):
        pose_row = []
        for landmark in wrists_id:
            start = (point-window_length+1)
            pose_row += [accelerations[point,0,landmark], # Single point acceleration
                         np.mean(accelerations[start if start >= 0 else 0:point+1,0,landmark]), # Mean of <window_length> accelerations
                         np.average(accelerations[start if start >= 0 else 0:point+1,0,landmark],weights=pose_visibility[start if start >= 0 else 0:point+1,landmark,0])] # Mean of <window_length> accelerations, weighted with visibility
            pose_row += common_columns[landmark]
        pose_X += [pose_row]
    
    pose_X = pd.DataFrame(pose_X)
    pose_reaction_class = model.predict(pose_X)[0]
    pose_reaction_prob = model.predict_proba(pose_X)[0]
    print(pose_reaction_class, pose_reaction_prob)

with open(os.path.join(directories["support_files"],'pose.pkl'), 'rb') as f:
    model = pickle.load(f)

detect_reaction(0,window_length,model)