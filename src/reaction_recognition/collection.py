import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from data_structures import Reaction
from utils import compute_acceleration

def detect_landmarks(source,pose_file,window_length,label):

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions

    count = 0
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

                # Face Detections
                '''mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )'''

                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )

                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    '''positions += [[
                        [pose[15].x, pose[15].y, pose[15].z],
                        [pose[16].x, pose[16].y, pose[16].z]
                    ]]'''
                    positions += [[[landmark.x, landmark.y, landmark.z] for landmark in pose]]
                    visibility += [[[landmark.visibility] for landmark in pose]]
                    
                    # Extract Face landmarks
                    '''face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                    
                    # Append class name 
                    face_row.insert(0, label)
                    
                    export_to_csv(files["face"],"a",face_row)'''
                    
                except NameError:
                    print(NameError)
                    pass
                        
                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
                # Frame block
                if count < 10:
                    count += 1
                else:
                    break
    
    cap.release()
    cv2.destroyAllWindows()

    # Hypotesis: (frame,point,dims) -> (point,dims,frame)
    positions = np.array(positions)
    visibility = np.array(visibility)
    
    # Plot the positions over time
    '''plt.figure(figsize=(12, 9))
    plt.plot(positions[:,15,0]), label="left_x")
    plt.plot(positions[:,15,1]), label="left_y")
    plt.plot(positions[:,15,2]), label="left_z")
    plt.plot(positions[:,16,0]), label="right_x")
    plt.plot(positions[:,16,1]), label="right_y")
    plt.plot(positions[:,16,2]), label="right_z")
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

    '''
    # Left hand acceleration + visibility
    left_data = np.array([[accelerations[point,0,15],visibility[point,15,0]] for point in range(acceleration_shape[0])])
    # Right hand acceleration + visibility
    right_data = np.array([[accelerations[point,0,16],visibility[point,16,0]] for point in range(acceleration_shape[0])])
    print(np.shape(left_data),np.shape(right_data))
    # Pose data
    pose_row = [label] + list(left_data.flatten()) + list(right_data.flatten())
    '''

    '''row [label,
    left_acceleration_mean,left_visibility_mean,left_(acceleration*visibility)_element_mean,
    right_acceleration_mean,right_visibility_mean,right_(acceleration*visibility)_element_mean]'''
    common_columns = {}
    for landmark in [15,16]:
        common_columns[landmark] = [np.mean(accelerations[:,0,landmark]), # Mean of all accelerations
                                    np.average(accelerations[:,0,landmark],weights=visibility[:,landmark,0])] # Mean of all accelerations, weighted with visibility

    for point in range(np.shape(positions)[0]):
        pose_row = [label] # maybe window_length
        for landmark in [15,16]:
            start = (point-window_length+1)
            pose_row += [accelerations[point,0,landmark], # Single point acceleration
                         np.mean(accelerations[start if start >= 0 else 0:point+1,0,landmark]), # Mean of <window_length> accelerations
                         np.average(accelerations[start if start >= 0 else 0:point+1,0,landmark],weights=visibility[start if start >= 0 else 0:point+1,landmark,0])] # Mean of <window_length> accelerations, weighted with visibility
            pose_row += common_columns[landmark]
        export_to_csv(pose_file,"a",pose_row)
    print(visibility[:,15,0])

def export_to_csv(file,mode,row):
    with open(file, mode=mode, newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)

fromWebCam = True
label = "hi"
window_length = 5
files = {
    "pose": "pose.csv",
    "face": "face.csv",
}
newFile = {
    "face": True,
    "acceleration": True
}
directory = "./samples/"

'''a = [-1.4696402549743663,-1.8762204051017772,-1.9387177824974076,-1.7575988173484813,-1.7006577253341686,-1.4998279213905346]
b = [0.99557209,0.99437881,0.993406,0.99242508,0.99085259,0.99040836]
print(a[:window_length])
print(np.mean(a[:window_length]),np.average(a[:window_length],weights=b[:window_length]))
print(a[1:window_length+1])
print(np.mean(a[1:window_length+1]),np.average(a[1:window_length+1],weights=b[1:window_length+1]))
'''
if newFile["acceleration"]:
    landmarks = ["class"]
    for hand in ["l","r"]:
        landmarks += [f"{hand}_a",f"{hand}_wm_a",f"{hand}_wm_av",f"{hand}_m_a",f"{hand}_m_av"]
    
    export_to_csv(files["pose"],"w",landmarks)

if fromWebCam:
    detect_landmarks(0,files["pose"],window_length,label)
else:
    # List of file in a directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for file in files:
        detect_landmarks(os.path.join(directory, file),files["pose"],window_length,os.path.splitext(file)[0])