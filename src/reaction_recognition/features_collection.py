import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import matplotlib.pyplot as plt
import os

from data_structures import directories,features_files,wrists_id,Reaction,features_directories_correspondences,window_length
from utils import compute_acceleration,export_to_csv

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
    
    #Face header
    '''if output_files["face"].new:
        count = 0
        landmarks = ["class"]
        for eyebrow in ["l","r"]:
            landmarks += [f"{eyebrow}_c",f"{eyebrow}_w_c",f"{eyebrow}_w_c"]
        export_to_csv(os.path.join(directories["support_files"],output_files["face"].name),"w",landmarks)'''

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
            else:
                break
    
    cap.release()
    cv2.destroyAllWindows()

    # Hypotesis: (frame,point,dims) -> (point,dims,frame)
    positions = np.array(positions)
    visibility = np.array(visibility)
    
    # Plot the positions over time
    '''plt.figure(figsize=(12, 9))
    plt.plot(positions[:,15,0], label="left_x")
    plt.plot(positions[:,15,1], label="left_y")
    plt.plot(positions[:,15,2], label="left_z")
    plt.plot(positions[:,16,0], label="right_x")
    plt.plot(positions[:,16,1], label="right_y")
    plt.plot(positions[:,16,2], label="right_z")
    plt.legend()
    plt.grid(True)
    plt.show()'''

    accelerations = compute_acceleration(positions)
    
    # Plot the acceleration over time
    '''plt.figure(figsize=(12, 9))
    plt.plot(np.multiply(accelerations[:,0,15],visibility[:,15,0]), label="left")
    plt.plot(np.multiply(accelerations[:,0,16],visibility[:,16,0]), label="right")
    plt.legend()
    plt.grid(True)
    plt.show()'''

    
    '''accelerations_shape = np.shape(accelerations)
    # Left wrist acceleration + visibility
    left_data = np.array([[accelerations[point,0,15],visibility[point,15,0]] for point in range(accelerations_shape[0])])
    # Right wrist acceleration + visibility
    right_data = np.array([[accelerations[point,0,16],visibility[point,16,0]] for point in range(accelerations_shape[0])])
    print(np.shape(left_data),np.shape(right_data))
    # Pose data
    pose_row = [label] + list(left_data.flatten()) + list(right_data.flatten())'''

    common_columns = {}
    for landmark in wrists_id:
        common_columns[landmark] = [np.mean(accelerations[:,0,landmark]), # Mean of all accelerations
                                    np.average(accelerations[:,0,landmark],weights=visibility[:,landmark,0])] # Mean of all accelerations, weighted with visibility

    for point in range(np.shape(positions)[0]):
        pose_row = [label] # maybe window_length
        for landmark in wrists_id:
            start = (point-window_length+1)
            pose_row += [accelerations[point,0,landmark], # Single point acceleration
                         np.mean(accelerations[start if start >= 0 else 0:point+1,0,landmark]), # Mean of <window_length> accelerations
                         np.average(accelerations[start if start >= 0 else 0:point+1,0,landmark],weights=visibility[start if start >= 0 else 0:point+1,landmark,0])] # Mean of <window_length> accelerations, weighted with visibility
            pose_row += common_columns[landmark]
        export_to_csv(os.path.join(directories["support_files"],output_files["pose"]["name"]),"a",pose_row)

fromWebCam = False
label = "hi"

to_initialize = False
for file in features_files:
    if features_files[file]["new"]:
        to_initialize = True
        break

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