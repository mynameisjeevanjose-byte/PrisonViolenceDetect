import cv2
import mediapipe as mp
import numpy as np
import os

print("Loading MediaPipe...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

SEQUENCE_LENGTH = 10
X_data = []
y_data = []

# Map labels to folder names
DATASET_DIRS = {
    0: 'dataset/normal',
    1: 'dataset/brawl',
    2: 'dataset/fall'
}

basedir = os.path.abspath(os.path.dirname(__file__))

# Ensure folders exist
for d in DATASET_DIRS.values():
    os.makedirs(os.path.join(basedir, d), exist_ok=True)

print("\n--- AUTOMATED INTERNET DATASET PROCESSOR ---")
print("1. Download fight/fall/normal videos (.mp4, .avi).")
print("2. Place them in the correct folders inside the 'dataset/' directory.")
print("--------------------------------------------\n")

for label, folder in DATASET_DIRS.items():
    folder_path = os.path.join(basedir, folder)
    videos = [v for v in os.listdir(folder_path) if v.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        print(f"Skipping {folder} - No videos found.")
        continue
        
    print(f"\nProcessing {len(videos)} videos in {folder} (Class {label})...")
    
    for video_file in videos:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        sequence = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame_count += 1
            # Process every 5th frame to perfectly match the live app (~6 FPS / 1.6s rhythm)
            if frame_count % 5 != 0: continue
                
            frame = cv2.resize(frame, (640, 480))
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                cx = (landmarks[23].x + landmarks[24].x) / 2
                cy = (landmarks[23].y + landmarks[24].y) / 2
                cz = (landmarks[23].z + landmarks[24].z) / 2
                
                current_pose = [val for res in landmarks for val in (res.x-cx, res.y-cy, res.z-cz, res.visibility)]
                sequence.append(current_pose)
                
                if len(sequence) == SEQUENCE_LENGTH:
                    X_data.append(sequence.copy())
                    y_data.append(label)
                    sequence = sequence[5:] # Overlap sequences by 50% for maximum data extraction
                    
        cap.release()
        print(f" -> Finished {video_file}. Total sequences saved so far: {len(X_data)}")

if X_data:
    X_save = np.concatenate((np.load('X_data.npy'), X_data)) if os.path.exists('X_data.npy') else np.array(X_data)
    y_save = np.concatenate((np.load('y_data.npy'), y_data)) if os.path.exists('y_data.npy') else np.array(y_data)
    np.save('X_data.npy', X_save)
    np.save('y_data.npy', y_save)
    print(f"\nSUCCESS! {len(X_data)} sequences appended to your dataset files.")
    print("You can now run train_lstm_model.py!")
else:
    print("\nNo data extracted. Make sure you put video files into the dataset folders.")